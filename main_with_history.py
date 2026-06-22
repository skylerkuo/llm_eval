from langchain_core.messages import HumanMessage, AIMessage
from llm_utils import get_session_history, MAX_HISTORY
from decompose_llm import decompose_llm_infer
from answer_llm import answer_llm_infer
from clarify_llm import clarify_llm_infer
from front_llm_v2 import front_llm_infer 
from modify_llm import modify_llm_infer
# from vision_qa_llm import vision_qa_llm_infer

def main():
    session_id = "default"
    history = get_session_history(session_id)

    while True:
        user_input = input("input: ").strip()
        if not user_input:
            continue

        print(f"=== history ({len(history.messages)} 筆) ===")
        for msg in history.messages:
            print(f"  [{msg.type}] {msg.content}")
        print("===")

        result = front_llm_infer(user_input)
        print(f"LLM type: {result}")
        llm_type = result.get("type")
        response_text = None

        if llm_type == "plan":
            selected_name = result.get("selected_name")
            response = decompose_llm_infer(user_input)
            response_text = response.get("answer", "（無回答）")
            action_steps = response.get("steps", [])
            print(response_text)
            print(action_steps)
        elif llm_type == "env_query":
            response_text = "[FOCUS]"
            print(response_text)
        elif llm_type == "clarify":
            selected_name = result.get("selected_name")
            response = clarify_llm_infer(
                user_input,
                selected_name=selected_name
            )
            response_text = response.get("answer", "（無回答）")
            print(f"clarify: {response_text}")
        elif llm_type == "answer":
            response = answer_llm_infer(user_input)
            response_text = response.get("answer", "（無回答）")
            print(f"Answer: {response_text}")

        elif llm_type == "modify":
            response = modify_llm_infer(user_input)
            response_text = response.get("answer", "（無回答）")
            response_add = response.get("add", "（無回答）")
            print(f"Answer: {response_text}")
            print(f"Add: {response_add}")
        else:
            print("Unknown type")
            response_text = ""

        if response_text:
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=response_text))

main()