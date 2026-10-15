"""意图识别（二分类：校园咨询 / 通用闲聊）。

按优先级降级：
    1. bert-base-chinese 微调模型（首选，model/bert_intent 下若有分类头）
    2. sklearn TF-IDF + 人工特征（次选，需与训练脚本共用 extract_features）
    3. 关键词规则（兜底，保证服务永远可用）

注意：特征函数已统一到 src/common/intent_features.py，
训练脚本与这里必须共用，否则会出现特征漂移导致误判。
"""
import json
import os
import pickle
import threading
from typing import List, Optional

import numpy as np

from config import INTENT_MODEL_CONFIG
from ..common.assistant_intent import get_assistant_reply
from ..common.intent_features import (
    FEATURE_VERSION,
    GENERAL_KEYWORDS,
    ID_TO_LABEL,
    INTENT_CHITCHAT,
    INTENT_CONSULT,
    PROFESSIONAL_KEYWORDS,
    extract_features,
)


class IntentClassifier:
    def __init__(self):
        self.model_path = INTENT_MODEL_CONFIG["model_path"]
        self.device = INTENT_MODEL_CONFIG.get("device", "cpu")
        self.confidence_threshold = INTENT_MODEL_CONFIG.get("confidence_threshold", 0.6)
        self.preferred = INTENT_MODEL_CONFIG.get("backend", "auto")

        self.backend: Optional[str] = None
        self.torch = None
        self.tokenizer = None
        self.bert_model = None
        self.sklearn_model = None
        self.vectorizer = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    # ---------- 加载 ----------

    def _load(self):
        # 这里不能用"锁外看一眼 _loaded 就返回"的双重检查：_loaded 是在
        # 真正加载**之前**置位的（置位在前是为了加载失败也不反复重试），
        # 于是后到者会看到 _loaded=True 却 backend=None，静默走关键词规则，
        # 把咨询问题判成闲聊、直接跳过检索。所以必须无条件拿锁排队。
        with self._load_lock:
            if not self._loaded:
                self._loaded = True
                self._load_backend()

    def _load_backend(self):
        loaders = {
            "bert": (self._try_load_bert, "使用 bert-base-chinese 微调模型"),
            "sklearn": (self._try_load_sklearn, "使用 sklearn TF-IDF 模型"),
        }

        order = ["bert", "sklearn"] if self.preferred == "auto" else [self.preferred]
        for name in order:
            loader, message = loaders.get(name, (None, ""))
            if loader and loader():
                self.backend = name
                print(f"[IntentClassifier] {message}")
                return

        self.backend = "rules"
        detail = f"（{self._load_error}）" if self._load_error else ""
        print(f"[IntentClassifier] 无可用模型{detail}，降级为关键词规则")

    @property
    def is_loaded(self) -> bool:
        """只读状态，不触发加载（供 /api/health 用）。

        注意 backend 可能是 "rules"（降级兜底），那也是"加载完了"的终态。
        """
        return self.backend is not None

    def _classifier_num_labels(self) -> Optional[int]:
        """读取分类头的类别数。

        直接看权重形状最可靠：config.json 里的 num_labels 可能缺失，
        而分类头是否真的存在、是几分类，只有权重能说明。
        """
        safetensors_file = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(safetensors_file):
            try:
                import struct

                with open(safetensors_file, "rb") as f:
                    header_len = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(header_len))
                weight = header.get("classifier.weight")
                if weight:
                    return int(weight["shape"][0])
            except Exception as e:
                print(f"[IntentClassifier] 读取 safetensors 头失败: {e}")

        config_file = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    return json.load(f).get("num_labels")
            except Exception:
                pass
        return None

    def _try_load_bert(self) -> bool:
        weight_file = os.path.join(self.model_path, "pytorch_model.bin")
        if not (os.path.exists(weight_file)
                or os.path.exists(os.path.join(self.model_path, "model.safetensors"))):
            return False

        # 没有分类头的目录（例如直接放了一份 bert-base-chinese）不能当分类器用，
        # 否则会静默加载随机初始化的分类头，预测结果等同于瞎猜。
        if self._classifier_num_labels() != 2:
            self._load_error = (
                "bert_intent 目录下不是二分类模型，"
                "请先运行 scripts/train_intent_classifier.py"
            )
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, num_labels=2
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
            return True
        except Exception as e:
            self._load_error = f"加载 BERT 失败: {e}"
            return False

    def _try_load_sklearn(self) -> bool:
        model_file = os.path.join(self.model_path, "intent_model.pkl")
        vectorizer_file = os.path.join(self.model_path, "vectorizer.pkl")
        meta_file = os.path.join(self.model_path, "feature_meta.json")

        if not (os.path.exists(model_file) and os.path.exists(vectorizer_file)):
            return False

        try:
            meta = {}
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            # 特征版本不一致的旧模型直接弃用，避免"训练/推理特征不同"的静默误判
            if meta.get("feature_version") != FEATURE_VERSION:
                self._load_error = (
                    f"sklearn 模型特征版本为 {meta.get('feature_version')}，"
                    f"当前需要 {FEATURE_VERSION}，请重新运行 scripts/train_intent_sklearn.py"
                )
                return False

            with open(model_file, "rb") as f:
                self.sklearn_model = pickle.load(f)
            with open(vectorizer_file, "rb") as f:
                self.vectorizer = pickle.load(f)
            return True
        except Exception as e:
            self._load_error = f"加载 sklearn 模型失败: {e}"
            return False

    # ---------- 预测 ----------

    def classify(self, query: str) -> str:
        if not query or not query.strip():
            return INTENT_CHITCHAT

        # 明确询问助手能力的句式优先于模型，不能被“校园”等词带进咨询检索。
        if get_assistant_reply(query) is not None:
            return INTENT_CHITCHAT

        self._load()

        if self.backend == "bert":
            return self._classify_with_bert(query)
        if self.backend == "sklearn":
            return self._classify_with_sklearn(query)
        return self._classify_with_rules(query)

    def _classify_with_bert(self, query: str) -> str:
        try:
            inputs = self.tokenizer(
                query, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.inference_mode():
                logits = self.bert_model(**inputs).logits
                probs = self.torch.softmax(logits, dim=-1)[0]
            confidence, prediction = float(probs.max()), int(probs.argmax())

            if confidence < self.confidence_threshold:
                return self._classify_with_rules(query)
            return ID_TO_LABEL.get(prediction, INTENT_CONSULT)
        except Exception as e:
            print(f"[IntentClassifier] BERT 推理失败，降级规则: {e}")
            return self._classify_with_rules(query)

    def _classify_with_sklearn(self, query: str) -> str:
        from scipy.sparse import hstack

        try:
            tfidf = self.vectorizer.transform([query])
            manual = np.array([extract_features(query)], dtype=float)
            features = hstack([tfidf, manual])

            prediction = int(self.sklearn_model.predict(features)[0])
            proba = self.sklearn_model.predict_proba(features)[0]
            if float(max(proba)) < self.confidence_threshold:
                return self._classify_with_rules(query)
            return ID_TO_LABEL.get(prediction, INTENT_CONSULT)
        except Exception as e:
            print(f"[IntentClassifier] sklearn 推理失败，降级规则: {e}")
            return self._classify_with_rules(query)

    @staticmethod
    def _classify_with_rules(query: str) -> str:
        professional = sum(1 for kw in PROFESSIONAL_KEYWORDS if kw in query)
        general = sum(1 for kw in GENERAL_KEYWORDS if kw in query)

        if professional > general:
            return INTENT_CONSULT
        if professional == general and professional > 0:
            return INTENT_CONSULT
        return INTENT_CHITCHAT

    def classify_batch(self, queries: List[str]) -> List[str]:
        return [self.classify(q) for q in queries]
