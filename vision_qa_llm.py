from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image
import torch, json, re

MODEL_NAME = "google/gemma-4-E2B-it"
DEFAULT_IMAGE = "/home/gai/Desktop/GAI_secondyear_demo/llm_code_langchain_gemma4_ollama/received.jpg"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_NAME, device_map="cuda")

SYSTEM_PROMPT = """你是一個工業場域機器人助手。
觀察圖片，根據使用者問題回答物品的位置與數量，回答需具體且簡潔。
使用者用甚麼語言(英文或中文)，answer就要用相同語言回答。
你只能輸出 JSON，不可輸出其他文字。
輸出格式：
{
  "answer": "回答內容"
}"""

def parse_json_output(raw: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"answer": cleaned}

def vision_qa_llm_infer(user_text: str, image_path: str = DEFAULT_IMAGE) -> dict:

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                },
                {
                    "type": "text",
                    "text": f"問題：{user_text}",
                },
            ],
        },
    ]


    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text_prompt,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    raw_text = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )
    print(f"[VL Raw Output] {raw_text}")
    result = parse_json_output(raw_text)
    print(f"[Final Answer] {result}")
    return result


