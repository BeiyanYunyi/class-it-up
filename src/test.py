import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer

session = ort.InferenceSession("bert_model_quantized.onnx")

tokenizer: BertTokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")


def predict_onnx(comment):
    encoding = tokenizer(
        comment, padding="max_length", max_length=128, return_tensors="np"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    outputs = session.run(
        ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    prediction = np.argmax(outputs[0], axis=1)[0]

    return prediction


print(predict_onnx("这篇文章写得很好！"))
