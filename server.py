from fastapi import FastAPI
import numpy as np
from transformers import BertTokenizer
from pydantic import BaseModel
import onnxruntime as ort
import logging

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


class ReqBody(BaseModel):
    content: str


class Result(BaseModel):
    predicted_label: int


app = FastAPI()

# 加载模型
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


@app.post("/predict")
def predict(comment: ReqBody) -> Result:
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

    return Result(predicted_label=prediction)


logging.getLogger("uvicorn").info(
    f"smoke test, {predict(ReqBody(content='This is a test comment.'))}"
)
