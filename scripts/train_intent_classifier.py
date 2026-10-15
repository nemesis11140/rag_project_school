"""意图识别模型微调（bert-base-chinese）。

与 train_intent_sklearn.py 用完全相同的验证集划分，便于横向对比。

数据增强：data/高频问答对.csv 的问题作为「专业咨询」正样本加入训练集，
验证集只取 微调数据.json 的划分，避免自评虚高。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

from config import DATA_DIR, INTENT_MODEL_CONFIG, MODEL_DIR
from src.common.intent_features import (
    INTENT_CHITCHAT,
    INTENT_CONSULT,
    LABEL_TO_ID,
)

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
SEED = 42


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [i["text"] for i in data], [LABEL_TO_ID[i["label"]] for i in data]


def load_csv_questions():
    csv_path = os.path.join(DATA_DIR, "高频问答对.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "问题" not in df.columns:
        return []
    return [str(q).strip() for q in df["问题"] if str(q).strip() and str(q).strip().lower() != "nan"]


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions, truths = [], []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        truths.extend(batch["labels"].tolist())
    return predictions, truths


def main():
    torch.manual_seed(SEED)

    print("=" * 60)
    print("意图识别模型微调 (bert-base-chinese)")
    print("=" * 60)

    base_model_path = INTENT_MODEL_CONFIG["base_model_path"]
    output_path = INTENT_MODEL_CONFIG["model_path"]
    data_path = os.path.join(DATA_DIR, "微调数据.json")

    if not os.path.isdir(base_model_path):
        print(f"错误: 基础模型不存在 {base_model_path}")
        return 1
    if not os.path.exists(data_path):
        print(f"错误: 训练数据不存在 {data_path}")
        return 1

    print(f"\n基础模型: {base_model_path}")
    print(f"输出路径: {output_path}")

    print("\n[Step 1] 加载数据...")
    texts, labels = load_json_data(data_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    extra = load_csv_questions()
    train_texts += extra
    train_labels += [LABEL_TO_ID[INTENT_CONSULT]] * len(extra)

    print(f"  训练集: {len(train_texts)} 条（含增强 {len(extra)} 条）")
    print(f"  验证集: {len(val_texts)} 条")

    print("\n[Step 2] 加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(base_model_path)
    model = BertForSequenceClassification.from_pretrained(base_model_path, num_labels=2)
    model.to(device)

    train_loader = DataLoader(
        IntentDataset(train_texts, train_labels, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        IntentDataset(val_texts, val_labels, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False
    )

    print("\n[Step 3] 训练...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(total_steps // 10, 1), num_training_steps=total_steps
    )

    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        predictions, truths = evaluate(model, val_loader, device)
        accuracy = accuracy_score(truths, predictions)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={total_loss / len(train_loader):.4f}  "
              f"val_acc={accuracy:.4f}")

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            os.makedirs(output_path, exist_ok=True)
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"    -> 保存最佳模型 (val_acc={best_accuracy:.4f})")

    print("\n[Step 4] 评估最佳模型...")
    model = BertForSequenceClassification.from_pretrained(output_path).to(device)
    predictions, truths = evaluate(model, val_loader, device)

    accuracy = accuracy_score(truths, predictions)
    print(f"  验证准确率: {accuracy:.4f}")
    print()
    print(classification_report(
        truths, predictions,
        labels=[0, 1], target_names=[INTENT_CHITCHAT, INTENT_CONSULT], zero_division=0
    ))

    with open(os.path.join(output_path, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": os.path.basename(base_model_path),
            "train_size": len(train_texts),
            "val_accuracy": round(float(accuracy), 4),
        }, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"训练完成！模型保存在: {output_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
