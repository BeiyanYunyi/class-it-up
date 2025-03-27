import torch
import onnx
from transformers import BertTokenizer, BertForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, preprocess, QuantType

device = torch.device("cpu")

model_path = "./fine_tuned_bert"

# 加载模型
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# 导出 ONNX
dummy_input_ids = torch.randint(0, 20000, (1, 128)).to(device).to(torch.int64)
dummy_attention_mask = torch.ones((1, 128)).to(device).to(torch.int64)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    "bert_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=20,
)

print("ONNX 模型导出成功！")

# 加载 ONNX 模型
onnx_model = onnx.load("bert_model.onnx")

preprocess.quant_pre_process(
    input_model=onnx_model, output_model_path="bert_model_preprocess.onnx"
)

quantize_dynamic(
    model_input="bert_model_preprocess.onnx",
    model_output="bert_model_quantized.onnx",
    weight_type=QuantType.QInt8,
)

print("ONNX 量化完成，模型已保存！")
