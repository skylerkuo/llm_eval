import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from llm_utils import get_session_history, MAX_HISTORY
from decompose_llm import decompose_llm_infer
from answer_llm import answer_llm_infer
from clarify_llm import clarify_llm_infer
from front_llm_v2 import front_llm_infer
from modify_llm import modify_llm_infer
# from vision_qa_llm import vision_qa_llm_infer
from pc_client import client
import asyncio
from faster_whisper import WhisperModel
from action_llm import robot_action_llm_infer

whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

def transcribe_audio(audio_path: str) -> str:
    if audio_path is None:
        return ""
    segments, _ = whisper_model.transcribe(audio_path, language="zh",initial_prompt="以下是繁體中文的轉錄內容。")
    return "".join([seg.text for seg in segments])

async def predict(user_input, history):
    if not user_input or not user_input.strip():
        return history, "waiting..."

    result = front_llm_infer(user_input)
    llm_type = result.get("type")

    response_text = ""
    extra_info = ""

    if llm_type == "clarify":

        selected_name = result.get("selected_name")

        response = clarify_llm_infer(
            user_input,
            selected_name=selected_name
        )
        response_text = f"clarify：{response.get('answer', '')}"

    elif llm_type == "robot_action":
        selected_name = result.get("selected_name")

        response = robot_action_llm_infer(
            user_input,
            selected_name=selected_name
        )

        response_text = response.get("answer", "")
        actions = response.get("actions", [])

        print(response)

        if actions:
            extra_info = "### Robot Action\n\n"
            extra_info += f"**Selected skill:** `{response.get('selected_name', selected_name)}`\n\n"

            for i, action in enumerate(actions, start=1):
                name = action.get("name", "")
                args = action.get("args", {})

                extra_info += f"#### Action {i}\n"
                extra_info += f"- **name:** `{name}`\n"

                if isinstance(args, dict) and args:
                    extra_info += "- **args:**\n"
                    for k, v in args.items():
                        extra_info += f"  - `{k}`: `{v}`\n"
                elif isinstance(args, list) and args:
                    extra_info += "- **args:**\n"
                    for idx, v in enumerate(args):
                        extra_info += f"  - `{idx}`: `{v}`\n"
                else:
                    extra_info += "- **args:** `{}`\n"
        else:
            extra_info = "### Robot Action\n\n沒有產生可執行的 action。"

    elif llm_type == "answer":
        selected_name = result.get("selected_name")

        response = answer_llm_infer(
            user_input,
            selected_name=selected_name
        )
        response_text = response.get("answer", "")

    elif llm_type == "plan":
        selected_name = result.get("selected_name")

        response = decompose_llm_infer(user_input)

        response_text = response.get("answer", "（無回答）")
        action_steps = response.get("steps", [])
        print(response_text)
        print(action_steps)
        try:
            await client(msg=action_steps, img=False)
        except Exception as e:
            print(f"[WebSocket Error] {e}")

        if action_steps:
            extra_info = "### Task Planning：\n\n"
            for step in action_steps:
                if isinstance(step, dict):
                    step_id = step.get("id", "")
                    action  = step.get("action", "no action")
                    obj     = step.get("object", "no object")
                    pos     = step.get("position", "no position")

                    extra_info += (
                        f"\n\n### Step {step_id}\n"
                        f"- **Action**：`{action}`\n"
                        f"- **Object**：`{obj}`\n"
                        f"- **Position**：`{pos}`\n"
                    )
                else:
                    extra_info += f"- {step}\n"

    elif llm_type == "modify":
        response = modify_llm_infer(user_input)
        response_text = response.get("answer", "")
        add_content = response.get("add", "")
        if add_content:
            extra_info = f"### new rules：\n{add_content}"

    elif llm_type == "env_query":
        # await client(msg="", img=True)
        # response = vision_qa_llm_infer(user_input)
        # response_text = response.get("answer", "")
        response_text = "桌上有白色的板子和橙色的部件。"
        print("vision")

    else:
        response_text = "no class"

    langchain_history = get_session_history("default")
    langchain_history.add_message(HumanMessage(content=user_input))
    langchain_history.add_message(AIMessage(content=response_text))

    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": response_text})

    return history, extra_info


with gr.Blocks(title="子計畫三：語音引導學習") as demo:
    gr.Markdown("# 子計畫三：語音引導學習")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="communication window", height=400)

            msg = gr.Textbox(
                placeholder="輸入問題，或用下方麥克風錄音後自動填入...",
                show_label=False,
            )

            # 🎙️ 按一次開始錄音，再按一次停止 → 自動轉文字填入 msg
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙️ 點擊開始錄音 / 再點一次停止並自動轉文字",
            )

            with gr.Row():
                submit = gr.Button("send", variant="primary")
                clear  = gr.Button("clear")

        with gr.Column(scale=2):
            gr.Markdown("### task planning (Steps / Details)")
            info_panel = gr.Markdown("waiting...", elem_id="info_panel")

    # 停止錄音時自動觸發轉文字，結果填入 msg
    audio_input.stop_recording(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[msg],
    )

    submit.click(
        predict,
        inputs=[msg, chatbot],
        outputs=[chatbot, info_panel],
    ).then(lambda: "", None, [msg])

    msg.submit(
        predict,
        inputs=[msg, chatbot],
        outputs=[chatbot, info_panel],
    ).then(lambda: "", None, [msg])

    clear.click(
        lambda: ([], "waiting..."),
        None,
        [chatbot, info_panel],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7210, share=True)