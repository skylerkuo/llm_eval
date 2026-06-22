from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage

from llm_utils import get_session_history, MAX_HISTORY
from decompose_llm import decompose_llm_infer


def print_line() -> None:
    print("-" * 80)


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def print_plan_result(response: dict[str, Any]) -> None:
    """
    將 decompose_llm_infer() 的輸出結果印在終端機。
    預期 response 格式大致為：
    {
        "answer": "...",
        "steps": [
            {"action": "moveto", "object": "sugar_box"},
            ...
        ],
        "skill_name": "move"
    }
    """

    answer = response.get("answer", "（無回答）")
    steps = response.get("steps", [])
    skill_name = (
        response.get("skill_name")
        or response.get("selected_name")
        or response.get("name")
        or "unknown"
    )

    print_line()
    print("LLM 回覆：")
    print(answer)
    print()

    print("Selected Skill：")
    print(skill_name)
    print()

    print("Task Planning：")

    if not steps:
        print("沒有產生任務規劃 steps。")
        print_line()
        return

    for i, step in enumerate(steps, start=1):
        print(f"\nStep {i}")

        if isinstance(step, dict):
            for key, value in step.items():
                print(f"  {key}: {format_value(value)}")
        else:
            print(f"  {step}")

    print_line()


def update_history(user_input: str, response_text: str) -> None:
    """
    保留原本的 LangChain session history。
    如果你不需要歷史紀錄，也可以整個函式刪掉。
    """
    history = get_session_history("default")

    history.add_message(HumanMessage(content=user_input))
    history.add_message(AIMessage(content=response_text))

    while len(history.messages) > MAX_HISTORY:
        history.messages.pop(0)


def run_once(user_input: str) -> None:
    """
    單次處理使用者文字指令。
    不經過 front_llm，永遠當作 plan 處理。
    """

    try:
        response = decompose_llm_infer(user_input)

        if not isinstance(response, dict):
            print_line()
            print("decompose_llm_infer() 回傳格式不是 dict：")
            print(response)
            print_line()
            return

        print_plan_result(response)

        response_text = response.get("answer", "")
        update_history(user_input, response_text)

    except KeyboardInterrupt:
        raise

    except Exception as e:
        print_line()
        print("[Error] 執行 decompose_llm_infer 時發生錯誤：")
        print(repr(e))
        print_line()


def main() -> None:
    print("Terminal Plan Mode")
    print("輸入自然語言指令後，會直接使用 decompose_llm_infer() 產生任務規劃。")
    print("輸入 exit / quit / q 離開。")
    print_line()

    while True:
        try:
            user_input = input("\n請輸入指令 > ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print("結束。")
                break

            run_once(user_input)

        except KeyboardInterrupt:
            print("\n結束。")
            break


if __name__ == "__main__":
    main()