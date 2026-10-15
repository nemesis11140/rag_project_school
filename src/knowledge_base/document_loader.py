"""文档读取 + 预处理。

流程：
    1. pdfplumber 逐页抽取文本层
    2. 文本过少的页面（扫描页）走 PaddleOCR 兜底（可选，未安装则跳过）
    3. 跨页重复行检测 -> 去除页眉 / 页脚 / 水印 / 书名
    4. 目录行、纯页码行过滤

为什么用"跨页重复"而不是正则白名单：
    旧实现用 `^[A-Za-z0-9\\s\\-_.]+$` 这类规则，会把课程编号、英文术语、
    成绩等级等正文一并删掉。页眉页脚/水印的本质特征是"在大量页面重复出现"，
    按重复率判定既准又不会误伤正文。
"""
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pdfplumber

from config import DOCUMENT_CONFIG

# 页眉页脚/水印判定：短行 + 出现在 >= 30% 的页面上（且至少 3 页）
REPEAT_LINE_MAX_CHARS = 60
REPEAT_MIN_PAGES = 3
REPEAT_PAGE_RATIO = 0.3

# 页码类
PAGE_NUMBER_PATTERNS = [
    r"^第\s*\d+\s*页$",
    r"^\d+\s*/\s*\d+$",
    r"^Page\s+\d+(\s+of\s+\d+)?$",
    r"^\d{1,4}$",
    r"^[-—–\s]*\d{1,4}[-—–\s]*$",
]

# 目录行：带点线引导或大量空格 + 结尾页码
TOC_LINE_PATTERN = re.compile(
    r"^.{2,60}?[\.·…\s]{2,}\s*\d{1,4}\s*$"
)
TOC_TITLES = {"目录", "目 录", "目　录", "contents", "table of contents"}


class DocumentLoader:
    def __init__(self, min_page_chars: int = None, ocr_fallback: Optional[bool] = None,
                 ocr_lang: str = None, ocr_dpi: int = None):
        self.min_page_chars = (
            DOCUMENT_CONFIG["min_page_chars"] if min_page_chars is None else min_page_chars
        )
        self.ocr_fallback = (
            DOCUMENT_CONFIG["ocr_fallback"] if ocr_fallback is None else ocr_fallback
        )
        self.ocr_lang = ocr_lang or DOCUMENT_CONFIG["ocr_lang"]
        self.ocr_dpi = ocr_dpi or DOCUMENT_CONFIG["ocr_dpi"]
        self._ocr_engine = None
        self._ocr_checked = False
        self.ocr_available = False

    # ---------- OCR ----------

    def _ensure_ocr(self):
        if self._ocr_checked:
            return
        self._ocr_checked = True

        if self.ocr_fallback is False:
            return

        # paddlepaddle 3.x 的 oneDNN 后端在部分 CPU 上会抛
        # "ConvertPirAttribute2RuntimeAttribute not support"，关掉 MKLDNN 可绕开
        os.environ.setdefault("FLAGS_use_mkldnn", "0")

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print(
                "[DocumentLoader] 未安装 PaddleOCR，扫描页无法识别。"
                "如需 OCR 兜底请执行: uv add paddleocr paddlepaddle"
            )
            return

        try:
            self._ocr_engine = self._build_ocr_engine(PaddleOCR)
            self.ocr_available = True
            print("[DocumentLoader] PaddleOCR 已就绪（仅用于文本层缺失的扫描页）")
        except Exception as e:
            print(f"[DocumentLoader] PaddleOCR 初始化失败，跳过 OCR: {e}")

    def _build_ocr_engine(self, paddle_ocr_cls):
        """兼容 PaddleOCR 2.x / 3.x 两套构造参数。"""
        try:
            # 3.x：参数改名 use_angle_cls -> use_textline_orientation，show_log 被移除
            return paddle_ocr_cls(
                lang=self.ocr_lang,
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=False,
            )
        except TypeError:
            # 2.x
            return paddle_ocr_cls(use_angle_cls=True, lang=self.ocr_lang, show_log=False)

    def _ocr_page(self, page) -> str:
        # 惰性初始化：PaddleOCR 首次构造要加载模型（CPU 上约 3 分钟），
        # 纯文本层文档根本用不到 OCR，不该为它付这个代价
        self._ensure_ocr()
        if not self.ocr_available:
            return ""
        try:
            import numpy as np

            image = np.array(page.to_image(resolution=self.ocr_dpi).original)
            # 3.x 用 predict()，2.x 用 ocr()
            if hasattr(self._ocr_engine, "predict"):
                results = self._ocr_engine.predict(image)
            else:
                results = self._ocr_engine.ocr(image, cls=True)
            return "\n".join(self._extract_ocr_texts(results))
        except Exception as e:
            print(f"[DocumentLoader] OCR 第 {page.page_number} 页失败: {e}")
            return ""

    @staticmethod
    def _extract_ocr_texts(results) -> List[str]:
        """兼容 3.x 的 rec_texts 结构与 2.x 的嵌套 [[box, (text, score)]] 结构。"""
        texts: List[str] = []

        for result in results or []:
            rec_texts = None
            if isinstance(result, dict):
                rec_texts = result.get("rec_texts")
            elif hasattr(result, "get"):
                try:
                    rec_texts = result.get("rec_texts")
                except Exception:
                    rec_texts = None

            if rec_texts:
                texts.extend(str(t) for t in rec_texts if str(t).strip())
                continue

            for item in result or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    info = item[1]
                    if isinstance(info, (list, tuple)) and info:
                        texts.append(str(info[0]))

        return texts

    # ---------- 行级清洗 ----------

    @staticmethod
    def _normalize_for_repeat(line: str) -> str:
        """去掉数字后比较，让 "第 12 页" 和 "第 13 页" 视为同一条页眉。"""
        return re.sub(r"\d+", "#", line.strip())

    @staticmethod
    def _is_page_number(line: str) -> bool:
        return any(re.match(p, line) for p in PAGE_NUMBER_PATTERNS)

    @staticmethod
    def _is_toc_line(line: str) -> bool:
        return bool(TOC_LINE_PATTERN.match(line)) or line.strip().lower() in TOC_TITLES

    def _find_repeated_lines(self, pages: List[Dict]) -> set:
        counter = Counter()
        for page in pages:
            seen_on_page = {
                self._normalize_for_repeat(line)
                for line in page["text"].split("\n")
                if line.strip() and len(line.strip()) <= REPEAT_LINE_MAX_CHARS
            }
            counter.update(seen_on_page)

        threshold = max(REPEAT_MIN_PAGES, int(len(pages) * REPEAT_PAGE_RATIO))
        repeated = {line for line, count in counter.items() if count >= threshold}
        if repeated:
            print(f"[DocumentLoader] 识别到 {len(repeated)} 条页眉/页脚/水印（跨页重复 >= {threshold} 页）")
        return repeated

    def _clean_page_text(self, text: str, repeated: set) -> str:
        kept = []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if self._is_page_number(line):
                continue
            if self._is_toc_line(line):
                continue
            if self._normalize_for_repeat(line) in repeated:
                continue
            kept.append(line)
        return "\n".join(kept)

    # ---------- 对外接口 ----------

    def load_pdf(self, file_path: str) -> Tuple[str, List[Dict]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        pages: List[Dict] = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""

                # 只在文本层确实过少时才碰 OCR（_ocr_page 内部按需初始化引擎）
                if len(text.strip()) < self.min_page_chars and self.ocr_fallback is not False:
                    ocr_text = self._ocr_page(page)
                    if len(ocr_text.strip()) > len(text.strip()):
                        print(f"[DocumentLoader] 第 {page_num} 页文本层过少，改用 OCR")
                        text = ocr_text

                pages.append({
                    "page_num": page_num,
                    "text": text,
                    "width": page.width,
                    "height": page.height,
                    "ocr": False
                })

        repeated = self._find_repeated_lines(pages)

        cleaned_pages = []
        for page in pages:
            cleaned = self._clean_page_text(page["text"], repeated)
            page["text"] = cleaned
            if cleaned.strip():
                cleaned_pages.append(page)

        combined = "\n".join(p["text"] for p in cleaned_pages)
        return combined, cleaned_pages

    def load_document(self, file_path: str) -> Tuple[str, List[Dict]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    # 兼容旧接口
    def clean_text(self, text: str) -> str:
        return self._clean_page_text(text, set())
