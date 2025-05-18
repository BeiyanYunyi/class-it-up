from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

tokenizer.save_pretrained("./fine_tuned_bert")
