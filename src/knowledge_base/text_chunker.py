"""中文递归文本切割器 + 父子分块。

切割流程：
    1. 用 RecursiveTextSplitter 按中英文标点优先级递归切割全文 -> 父块
    2. 每个父块再用更小的 chunk_size 递归切割 -> 子块

检索时用子块（语义集中、召回准），送入 LLM 时用父块（上下文完整）。
"""
from typing import List, Dict, Any, Optional
from config import CHUNKING_CONFIG

# 分隔符按优先级从高到低排列，"" 表示退化为按字符硬切
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "…", "，", "、", " ", ""]


class RecursiveTextSplitter:
    """递归字符切割器。

    与 LangChain 的 RecursiveCharacterTextSplitter 思路一致：
    优先用高优先级分隔符保持语义完整，若切出的片段仍超长，
    则用更低优先级的分隔符继续递归切割，最后退化为定长滑窗。

    构造时即校验 chunk_overlap < chunk_size，保证滑窗步长 >= 1，
    因此 split_text 一定终止。
    """

    def __init__(self, chunk_size: int, chunk_overlap: int,
                 separators: Optional[List[str]] = None):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正数，当前为 {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap 不能为负数，当前为 {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})，"
                "否则滑窗步长 <= 0 会导致死循环"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        chunks = self._split(text, list(self.separators))
        # 兜底：保证任何返回片段都不超过 chunk_size
        result = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                result.extend(self._hard_split(chunk))
            else:
                result.append(chunk)
        return [c for c in result if c.strip()]

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """用 separators[0] 切分，超长片段交给剩余分隔符继续递归。"""
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        new_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        pieces = self._split_with_separator(text, separator)

        final_chunks: List[str] = []
        good_splits: List[str] = []

        for piece in pieces:
            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
                continue

            # 遇到超长片段，先结算已积累的短片段
            if good_splits:
                final_chunks.extend(self._merge_splits(good_splits))
                good_splits = []

            if new_separators:
                final_chunks.extend(self._split(piece, new_separators))
            else:
                final_chunks.extend(self._hard_split(piece))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits))

        return final_chunks

    @staticmethod
    def _split_with_separator(text: str, separator: str) -> List[str]:
        """切分并把分隔符保留在前一个片段末尾，避免标点丢失。"""
        if separator == "":
            return list(text)

        parts = text.split(separator)
        pieces = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                pieces.append(part + separator)
            else:
                pieces.append(part)
        return [p for p in pieces if p != ""]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """把短片段贪心合并到接近 chunk_size，相邻块之间保留 overlap。"""
        chunks: List[str] = []
        buffer = ""

        for piece in splits:
            if buffer and len(buffer) + len(piece) > self.chunk_size:
                chunks.append(buffer.strip())
                buffer = self._overlap_tail(buffer) + piece
            elif not buffer and len(piece) > self.chunk_size:
                # 单片就超长，交给上层兜底硬切
                chunks.append(piece.strip())
            else:
                buffer += piece

        if buffer.strip():
            chunks.append(buffer.strip())

        return chunks

    def _overlap_tail(self, text: str) -> str:
        """取尾部 overlap 长度的文本，尽量从分隔符之后开始，避免切断句子。"""
        if self.chunk_overlap <= 0 or len(text) <= self.chunk_overlap:
            return ""
        tail = text[-self.chunk_overlap:]
        for sep in self.separators:
            if sep and sep in tail:
                idx = tail.find(sep)
                if idx != -1 and idx + len(sep) < len(tail):
                    return tail[idx + len(sep):]
        return tail

    def _hard_split(self, text: str) -> List[str]:
        """定长滑窗硬切。步长 = chunk_size - chunk_overlap >= 1，必然终止。"""
        step = self.chunk_size - self.chunk_overlap
        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= length:
                break
            start += step

        return chunks


class ParentChildChunker:
    def __init__(self, parent_chunk_size: int = None, parent_chunk_overlap: int = None,
                 child_chunk_size: int = None, child_chunk_overlap: int = None,
                 min_child_chunk_size: int = None):
        self.parent_chunk_size = parent_chunk_size or CHUNKING_CONFIG["parent_chunk_size"]
        self.parent_chunk_overlap = (
            CHUNKING_CONFIG["parent_chunk_overlap"] if parent_chunk_overlap is None else parent_chunk_overlap
        )
        self.child_chunk_size = child_chunk_size or CHUNKING_CONFIG["child_chunk_size"]
        self.child_chunk_overlap = (
            CHUNKING_CONFIG["child_chunk_overlap"] if child_chunk_overlap is None else child_chunk_overlap
        )
        self.min_child_chunk_size = (
            CHUNKING_CONFIG.get("min_child_chunk_size", 20)
            if min_child_chunk_size is None else min_child_chunk_size
        )

        self.parent_splitter = RecursiveTextSplitter(
            self.parent_chunk_size, self.parent_chunk_overlap
        )
        self.child_splitter = RecursiveTextSplitter(
            self.child_chunk_size, self.child_chunk_overlap
        )

    def split_text(self, text: str, document_name: str = "") -> List[Dict[str, Any]]:
        parent_chunks = self.parent_splitter.split_text(text)

        result: List[Dict[str, Any]] = []
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            child_chunks = self.child_splitter.split_text(parent_chunk)

            # 过滤过短的碎片，但父块只有一个子块时保留（否则整块丢失）
            if len(child_chunks) > 1:
                child_chunks = [
                    c for c in child_chunks if len(c) >= self.min_child_chunk_size
                ]
            if not child_chunks:
                continue

            for child_idx, child_chunk in enumerate(child_chunks):
                result.append({
                    "parent_text": parent_chunk,
                    "parent_index": parent_idx,
                    "child_text": child_chunk,
                    "child_index": child_idx,
                    "document_name": document_name,
                    "metadata": {
                        "document_name": document_name,
                        "parent_index": parent_idx,
                        "child_index": child_idx,
                        "parent_text": parent_chunk,
                        "total_parent_chunks": len(parent_chunks),
                        "total_child_chunks_in_parent": len(child_chunks)
                    }
                })

        return result

    def split_documents(self, texts: List[str], document_names: List[str]) -> List[Dict[str, Any]]:
        all_chunks = []
        for text, doc_name in zip(texts, document_names):
            chunks = self.split_text(text, doc_name)
            all_chunks.extend(chunks)
        return all_chunks
