from fastapi import FastAPI
from transformers import BertTokenizer
import torch
from pydantic import BaseModel


class Item(BaseModel):
    content: str


app = FastAPI()
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)

# 加载模型
model = torch.jit.load("./fine_tuned_bert/bert_torchscript.pt").to(device)
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")


@app.post("/predict")
def predict(comment: Item):
    encoding = tokenizer(
        comment.content,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding[
        "attention_mask"
    ].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(logits, dim=1).item()

    return {"predicted_label": prediction}


# 运行服务器
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
