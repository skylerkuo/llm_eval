from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from model_load_4b_4bit import chat_model
from gemma_parser import strip_gemma          # ← NEW
from llm_utils import (
    load_markdown_items,
    get_item_by_name,
    build_skill_catalog,
)

MAX_NEW_TOKENS = 200
BASE_DIR = "skill_data"

# =========================
# Prompts
# =========================

select_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是任務技能選擇助手。
你只能輸出合法 JSON，不可以輸出任何 JSON 以外的文字。

輸出格式：
{{
  "skill_name": "技能name"
}}"""),
    ("human", """技能清單:
{catalog}

現在任務:
{user_text}

請選一個最相關的skill"""),
])

decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是任務分解的機器人。

技能說明:
{skill_content}

注意: 新增規則是使用者後面新增的任務規則，請優先考慮使用者新增的任務規則再考慮範例。

輸出格式：
     
{{
  "steps": [
    {{ "id":"1" , "action": "first action name" , "object": "first object name" , "position": "first position" }},
    {{ "id":"2" , "action": "second action name" , "object": "second object name" , "position": "second position"}},
    {{ "id":"3" , "action": "third action name" , "object": "third object name" , "position": "third position"}}
  ],
  "answer": "(簡短回覆使用者你會完成什麼任務)"
}}
     
position如果沒有就輸出no
     
記得給使用者answer回覆
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。

id: 為每個步驟的編號，從1開始遞增。

只能輸出 JSON。"""),
    ("human", "{user_text}"),
])

json_parser = JsonOutputParser()

# =========================
# Chains (LCEL)
# =========================

select_chain   = select_prompt   | chat_model | strip_gemma | json_parser  # ← fixed
decompose_chain = decompose_prompt | chat_model | strip_gemma | json_parser  # ← fixed


# =========================
# Main
# =========================

def decompose_llm_infer(user_text: str, skill_name: str | None = None) -> dict:
    skills = load_markdown_items(BASE_DIR)

    if skill_name is None:
        catalog = build_skill_catalog(skills)
        selected = select_chain.invoke({"catalog": catalog, "user_text": user_text})
        skill_name = selected["skill_name"]

    skill = get_item_by_name(skills, skill_name)

    result = decompose_chain.invoke({
        "skill_content": skill["content"],
        "user_text": user_text,
    })

    result["skill_name"] = skill_name
    return result

