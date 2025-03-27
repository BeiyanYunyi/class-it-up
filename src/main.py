import torch
from transformers import BertTokenizer, BertForSequenceClassification

# model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")
model.to(device)


def predict_single_comment(comment, model, tokenizer, max_len=128):
    model.eval()
    encoding = tokenizer(
        comment,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction  # 0 表示其他用户，1 表示目标用户


# 测试单条评论
comment = r"联系你就是试图和你好好谈谈 还可以做个朋友。你一个劲儿的在这里乱解读。你有本事打个电话，见个面。心平气和聊聊。人家只是想和你不至于连朋友做不了。而你一直在诋毁他。你不要把对方对你的最后的好感和好意全部泯灭了。"
predicted_label = predict_single_comment(comment, model, tokenizer)
print(f"Predicted Label: {predicted_label}")


def main():
    print("Hello from censor!")


if __name__ == "__main__":
    main()
