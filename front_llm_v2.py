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
    ("system", """你是一個機器人檢索助手。
只能輸出 JSON，要找出核問題最相關的檔案，讓後續機器人能夠跟著檔案內容回答問題、釐清、行動、任務規劃、修改SOP。

{{
  "selected_name": "文件name"
}}"""),
    ("human", """知識文件：
{catalog}

問題：
{user_text}

請選最相關文件"""),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一個工業場域機器人。
根據提供的知識和現在使用者的指令選擇是哪一個種類，種類有 plan or env_query or clarify or answer or modify or robot_action。

輸出 JSON：
{{
  "type": "plan or env_query or clarify or answer or modify or robot_action"
}}

只能輸出 JSON。"""),
    ("human", """知識文件：

name: {name}
description: {description}

content:
{content}

問題：
{user_text}"""),
])

json_parser = JsonOutputParser()

# =========================
# Chains
# =========================

select_chain = select_prompt | chat_model | strip_gemma | json_parser
answer_chain = answer_prompt | chat_model | strip_gemma | json_parser


# =========================
# Main
# =========================

def front_llm_infer(user_text: str) -> dict:
    items = load_markdown_items(KNOWLEDGE_DIR)
    catalog = build_skill_catalog(items)

    # Step 1 — select md file
    selected = select_chain.invoke({
        "catalog": catalog,
        "user_text": user_text,
    })

    print(f"[front] selected: {selected}")

    selected_name = selected.get("selected_name")
    item = get_item_by_name(items, selected_name)

    # Step 2 — classify type
    result = answer_chain.invoke({
        "name": item["name"],
        "description": item["description"],
        "content": item["content"],
        "user_text": user_text,
    })

    # Step 3 — 把 front 選到的 md 資訊一起回傳
    result["selected_name"] = selected_name

    return result