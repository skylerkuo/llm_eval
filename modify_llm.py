import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from model_load_4b_4bit import chat_model
from gemma_parser import strip_gemma
from llm_utils import (
    load_markdown_items,
    get_item_by_name,
    build_skill_catalog,
)

KNOWLEDGE_DIR = "skill_data"

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
    ("system", """你是要負責任務規則補充的機器人。

你需要根據使用者的指令補充任務的描述，讓任務描述更完整。
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。

輸出格式：
{{
  "add": "寫上使用者需要補充的規則或SOP，50字以內",
  "answer": "簡短回覆使用者你會補充什麼"
}}

只能輸出 JSON。"""),
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
# Chains (LCEL)
# =========================

select_chain = select_prompt | chat_model | strip_gemma | json_parser
answer_chain = answer_prompt | chat_model | strip_gemma | json_parser


# =========================
# Main
# =========================

def modify_llm_infer(user_text: str) -> dict:
    items = load_markdown_items(KNOWLEDGE_DIR)
    catalog = build_skill_catalog(items)


    selected = select_chain.invoke({"catalog": catalog, "user_text": user_text})
    name = selected["selected_name"]
    item = get_item_by_name(items, name)

    result = answer_chain.invoke({
        "name":        item["name"],
        "description": item["description"],
        "content":     item["content"],
        "user_text":   user_text,
    })

    new_content = result.get("add", "")

    if new_content:
        with open(item["path"], "a", encoding="utf-8") as f:
            f.write(f"\n- {new_content}\n")

        print(f"[modify] 已新增規則到 {item['path']}")
        print(f"[modify] 新增規則：{new_content}")

    return result