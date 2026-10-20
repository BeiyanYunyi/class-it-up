import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

from data import load_dataframe


# 数据集定义
class CommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len=128):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.comments)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.comments[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(args):
    df = load_dataframe(args.input)
    if df.height < 2:
        raise ValueError("training requires at least two rows")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    # 划分数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["comment"].to_list(),
        df["label"].to_list(),
        test_size=args.test_size,
        random_state=args.seed,
    )

    tokenizer = BertTokenizer.from_pretrained(args.model)
    train_dataset = CommentDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = CommentDataset(val_texts, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2)
    device = torch.device(args.device if args.device != "auto" else (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    ))
    model.to(device)

    # 训练设置
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(model, train_loader, optimizer, criterion):
        model.train()
        total_loss, correct = 0, 0
        for batch in train_loader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
        return total_loss / len(train_loader), correct / len(train_dataset)

    def evaluate(model, val_loader, criterion):
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = (
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["label"].to(device),
                )
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
        return total_loss / len(val_loader), correct / len(val_dataset)

    # 训练循环
    epochs = args.epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

    # 保存模型
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model and tokenizer saved to {args.output}")
