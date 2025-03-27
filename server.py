from fastapi import FastAPI
import numpy as np
from transformers import BertTokenizer
import torch
from pydantic import BaseModel
import onnxruntime as ort

session = ort.InferenceSession(
    "bert_model_quantized.onnx",
    providers=["CPUExecutionProvider"],  # CPU 部署
    sess_options=ort.SessionOptions(),
)

# 启用内存优化（减少 GPU/CPU 内存占用）
session.set_providers(
    ["CPUExecutionProvider"],
    provider_options=[{"arena_extend_strategy": "kSameAsRequested"}],
)


class Item(BaseModel):
    content: str


app = FastAPI()
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

# 加载模型
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")


@app.post("/predict")
def predict(comment: Item):
    encoding = tokenizer(
        comment.content,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="np",
    )
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = session.run(
        ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    prediction = int(np.argmax(outputs[0], axis=1)[0])

    return {"predicted_label": prediction}


# 运行服务器
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
