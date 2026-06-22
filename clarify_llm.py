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

問題：
{user_text}

請選最相關文件"""),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一個工業場域機器人。
參照知識文件和對話紀錄和現在使用者的指令適時調整回答回覆使用者。
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。

輸出 JSON：
{{
  "answer": "回答"
}}"""),
    ("human", """知識文件：

name: {name}
description: {description}

content:
{content}

使用者問題：
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

def clarify_llm_infer(user_text: str, selected_name: str | None = None) -> dict:
    items = load_markdown_items(KNOWLEDGE_DIR)

    # 如果 front 已經選過 md，就直接使用
    if selected_name:
        print(f"[clarify] 使用 front 選到的 md: {selected_name}")
        name = selected_name

    # 如果沒有收到 selected_name，才重新選一次
    else:
        print("[clarify] 沒有收到 selected_name，重新選擇 md")

        catalog = build_skill_catalog(items)
        selected = select_chain.invoke({
            "catalog": catalog,
            "user_text": user_text
        })

        name = selected.get("selected_name")
        print(f"[clarify] selected: {name}")

    # 讀取對應 md
    item = get_item_by_name(items, name)

    # 產生澄清回答
    result = answer_chain.invoke({
        "name": item["name"],
        "description": item["description"],
        "content": item["content"],
        "user_text": user_text,
    })

    result["selected_name"] = name

    return result