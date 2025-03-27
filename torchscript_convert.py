import torch
from transformers import BertForSequenceClassification, BertTokenizer


class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits  # 仅返回 logits，不返回 dict


# 加载 fine-tuned 模型
model_path = "./fine_tuned_bert"
model = BertForSequenceClassification.from_pretrained(model_path)
wrapped_model = BertWrapper(model)
wrapped_model.eval()
tokenizer = BertTokenizer.from_pretrained(model_path)

# 切换到 eval 模式
wrapped_model.eval()
# 创建一个虚拟输入
dummy_text = "这篇文章写得很好！"
encoding = tokenizer(
    dummy_text,
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt",
)

dummy_input_ids = encoding["input_ids"]
dummy_attention_mask = encoding["attention_mask"]

# 将模型移动到 CPU 以便导出（可选）
device = torch.device("cpu")
wrapped_model.to(device)
dummy_input_ids, dummy_attention_mask = dummy_input_ids.to(
    device
), dummy_attention_mask.to(device)

traced_model = torch.jit.trace(wrapped_model, (dummy_input_ids, dummy_attention_mask))
traced_model.save("./fine_tuned_bert/bert_torchscript.pt")

print("TorchScript 模型已成功导出！")
