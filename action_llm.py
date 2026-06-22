from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from model_load_4b_4bit import chat_model
from gemma_parser import strip_gemma
from llm_utils import (
    load_markdown_items,
    get_item_by_name,
    build_skill_catalog,
)

KNOWLEDGE_DIR = "main_data"

# =========================
# Prompts
# =========================

select_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一個工業場域知識檢索助手。
你只能輸出 JSON。

{{
  "selected_name": "文件name"
}}"""),
    ("human", """知識文件：
{catalog}

使用者指令：
{user_text}

請選最相關文件"""),
])

robot_action_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一個工業場域機器人動作指令解析器。
你只能輸出 JSON。

請根據提供的檔案與使用者指令，輸出要執行的機器人動作，並產生一句要回覆使用者的 answer。

輸出 JSON 格式：

{{
  "type": "robot_action",
  "answer": "(簡短回覆使用者你會完成什麼動作)",
  "actions": [
    {{
      "name": "function_name",
      "args": {{}}
    }}
  ]
}}

記得給使用者answer回覆
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。
     
"""),
    ("human", """skill.md：

name: {name}
description: {description}

content:
{content}

使用者指令：
{user_text}"""),
])

json_parser = JsonOutputParser()

# =========================
# Chains
# =========================

select_chain = select_prompt | chat_model | strip_gemma | json_parser
robot_action_chain = robot_action_prompt | chat_model | strip_gemma | json_parser


# =========================
# Main
# =========================

def robot_action_llm_infer(user_text: str, selected_name: str | None = None) -> dict:
    items = load_markdown_items(KNOWLEDGE_DIR)

    if selected_name:
        print(f"[robot_action] 使用 front 選到的 md: {selected_name}")
        name = selected_name

    else:
        print("[robot_action] 沒有收到 selected_name，重新選擇 md")

        catalog = build_skill_catalog(items)
        selected = select_chain.invoke({
            "catalog": catalog,
            "user_text": user_text,
        })

        name = selected.get("selected_name")
        print(f"[robot_action] selected: {name}")

    item = get_item_by_name(items, name)

    result = robot_action_chain.invoke({
        "name": item["name"],
        "description": item["description"],
        "content": item["content"],
        "user_text": user_text,
    })

    result["selected_name"] = name

    return result