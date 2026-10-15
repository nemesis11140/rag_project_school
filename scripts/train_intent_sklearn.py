"""意图识别模型训练（sklearn: TF-IDF + 人工特征 + SVM）。

与线上推理共用 src/common/intent_features.extract_features，
并写出 feature_meta.json 供推理侧校验特征版本。

数据增强：把 data/高频问答对.csv 里的问题全部作为「专业咨询」正样本加入训练集
（都是真实的校园事务问法），但验证集只取 微调数据.json 的划分，避免自评虚高。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from config import DATA_DIR, MODEL_DIR
from src.common.intent_features import (
    FEATURE_VERSION,
    GENERAL_KEYWORDS,
    INTENT_CHITCHAT,
    INTENT_CONSULT,
    LABEL_TO_ID,
    PROFESSIONAL_KEYWORDS,
    extract_features,
)


def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [LABEL_TO_ID[item["label"]] for item in data]
    return texts, labels


def load_csv_questions():
    """高频问答对里的问题都是校园事务，作为额外正样本。"""
    csv_path = os.path.join(DATA_DIR, "高频问答对.csv")
    if not os.path.exists(csv_path):
        print(f"  [跳过] 未找到 {csv_path}")
        return []
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "问题" not in df.columns:
        return []
    return [str(q).strip() for q in df["问题"] if str(q).strip() and str(q).strip().lower() != "nan"]


def build_features(vectorizer, texts, fit=False):
    if fit:
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)
    manual = np.array([extract_features(t) for t in texts], dtype=float)
    return hstack([tfidf, manual]).tocsr()


def main():
    print("=" * 60)
    print("意图识别模型训练 (sklearn)")
    print("=" * 60)

    data_path = os.path.join(DATA_DIR, "微调数据.json")
    output_dir = os.path.join(MODEL_DIR, "bert_intent")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"错误: 训练数据不存在 {data_path}")
        return 1

    print("\n[Step 1] 加载数据...")
    texts, labels = load_json_data(data_path)
    print(f"  微调数据: {len(texts)} 条 "
          f"({INTENT_CHITCHAT} {labels.count(0)} / {INTENT_CONSULT} {labels.count(1)})")

    # 验证集只来自微调数据，保证评测干净
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    extra = load_csv_questions()
    if extra:
        train_texts = train_texts + extra
        train_labels = train_labels + [LABEL_TO_ID[INTENT_CONSULT]] * len(extra)
        print(f"  数据增强: +{len(extra)} 条高频问答(专业咨询) -> 训练集 {len(train_texts)} 条")
    print(f"  验证集: {len(val_texts)} 条")

    print("\n[Step 2] 特征提取...")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=10000)
    X_train = build_features(vectorizer, train_texts, fit=True)
    X_val = build_features(vectorizer, val_texts)
    print(f"  特征维度: {X_train.shape[1]}")

    print("\n[Step 3] 训练...")
    # 增强后正负样本不平衡（279:80），必须加 class_weight，
    # 否则闲聊类召回会塌掉（实测 0.40）
    model = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", class_weight="balanced")
    model.fit(X_train, train_labels)

    print("\n[Step 4] 评估（验证集为微调数据划分，无增强数据泄漏）...")
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(val_labels, val_predictions)
    print(f"  验证准确率: {accuracy:.4f}")
    print()
    print(classification_report(
        val_labels, val_predictions,
        labels=[0, 1], target_names=[INTENT_CHITCHAT, INTENT_CONSULT], zero_division=0
    ))

    print("[Step 5] 保存模型...")
    with open(os.path.join(output_dir, "intent_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(output_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_version": FEATURE_VERSION,
            "professional_keywords": len(PROFESSIONAL_KEYWORDS),
            "general_keywords": len(GENERAL_KEYWORDS),
            "train_size": len(train_texts),
            "val_accuracy": round(float(accuracy), 4),
        }, f, ensure_ascii=False, indent=2)
    print(f"  已保存到 {output_dir}")

    print("\n[Step 6] 冒烟测试...")
    test_queries = [
        ("今天天气怎么样？", INTENT_CHITCHAT),
        ("怎么煮鸡蛋？", INTENT_CHITCHAT),
        ("怎么减肥？", INTENT_CHITCHAT),
        ("食堂中午几点开门？", INTENT_CONSULT),
        ("图书馆的借书期限是多久？", INTENT_CONSULT),
        ("学费要交多少？", INTENT_CONSULT),
        ("怎么申请奖学金？", INTENT_CONSULT),
        ("学生证丢了去哪补办？", INTENT_CONSULT),
        ("明天会下雨吗？", INTENT_CHITCHAT),
        ("怎么做红烧肉？", INTENT_CHITCHAT),
    ]
    correct = 0
    for query, expected in test_queries:
        features = build_features(vectorizer, [query])
        proba = model.predict_proba(features)[0]
        prediction = int(model.predict(features)[0])
        intent = INTENT_CONSULT if prediction == 1 else INTENT_CHITCHAT
        mark = "✅" if intent == expected else "❌"
        correct += int(intent == expected)
        print(f"  {mark} {query:<24} -> {intent} (置信度 {max(proba):.2f}, 期望 {expected})")
    print(f"  冒烟测试: {correct}/{len(test_queries)}")

    print("\n" + "=" * 60)
    print(f"训练完成！模型保存在: {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
