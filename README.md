# LLM Eval

這是語言模型的系統，可以根據使用者指令去搜尋相關資料並回答問題或是任務分解。

## 檔案和資料夾介紹

### `main_data`

放 front llm 做任務分類和 clarify 和 answer 的所有資料。

### `skill_data`

放任務分解的相關指令。

### 主要程式檔案

- `answer_llm.py`
- `clarify_llm.py`
- `decompose_llm.py`
- `modify_llm.py`
- `vision_qa_llm`

檔案如其名。

### `front_llm_v2.py`

做一開始任務分類的。

### `gemma_parser.py`、`llm_utils.py`

幫 llm 做輸入輸出格式相關的處理。

### `model_load_4b_4bit.py`

匯入語言模型，如果要改要使用哪個語言模型，需要改這個檔案中的 model path。

### `gai_demo_main_voice.py`

最核心的程式，會呼叫 ui 介面可以對話。

### `llm_eval_main.py`

讓系統自己讀題庫 `llm_eval.jsonl`，並且將結果存成 `eval_result.jsonl`。

### `md_file_tutorial.md`

教 GPT 或人類寫 answer 或是 clarify 的 md 檔案。

### `pc_client.py`

幫 ui 的。

### 其他

其他都不重要。

## 下載語言模型

在 Hugging Face 下載模型：

https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF

也可以下載其他大小的，下載完後放到這個檔案夾裡面就好，並且記得改 `model_load_4b_4bit.py` 裡面的 model path，讓系統能夠讀到。

## 安裝 llama-cpp-python

`llama-cpp-python` 請安裝 CUDA 版本，可參考官方說明：

https://pypi.org/project/llama-cpp-python/

安裝格式如下：

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>
```

例如 CUDA 12.5：

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

若 `nvidia-smi` 顯示 CUDA 13.0，也建議可先使用 `cu125`，相容性通常較穩。

## 安裝環境

建議使用 Python 3.10 以上。

直接執行：

```bash
python gai_demo_main_voice.py
```

缺什麼補什麼

## 執行

### 1. 開 UI 對話

```bash
python gai_demo_main_voice.py
```

### 2. 語言模型評估

```bash
python llm_eval_main.py
```
