import re
import logging
from typing import List, Dict, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class TextChunk:
    content: str
    start_pos: int
    end_pos: int
    tokens: int
    chunk_id: int
    metadata: Dict[str, Union[str, int]] = field(default_factory=dict)


@dataclass
class SplitResult:
    chunks: List[TextChunk]
    total_chunks: int
    total_tokens: int
    original_length: int
    overlap_info: Dict[str, int]


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                pass

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return int(chinese_chars * 0.5 + (len(text) - chinese_chars) * 0.25)


class TextSplitter:
    def __init__(self, max_tokens: int = 3000, overlap_tokens: int = 200, encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = TokenCounter(encoding_name)
        self.split_patterns = [
            r'\n\s*\n\s*\n+', r'\n\s*\n', r'[。！？；]\s*\n', r'[。！？；]\s+',
            r'[，、]\s*\n', r'\n', r'[，、]\s+', r'\s+'
        ]

    def split_text(self, text: str) -> List[str]:
        return [c.content for c in self.split_text_advanced(text).chunks]

    def split_text_advanced(self, text: str) -> SplitResult:
        if not text or not text.strip():
            return SplitResult([], 0, 0, 0, {})
        text = self._preprocess(text)
        tokens = self.token_counter.count_tokens(text)
        if tokens <= self.max_tokens:
            chunk = TextChunk(text, 0, len(text), tokens, 0)
            return SplitResult([chunk], 1, tokens, len(text), {})
        chunks = self._split(text)
        chunks = self._merge_small(chunks)
        if self.overlap_tokens > 0:
            chunks = self._add_overlap(chunks)
        total_tokens = sum(c.tokens for c in chunks)
        return SplitResult(chunks, len(chunks), total_tokens, len(text), {"enabled": True})

    def split_text_with_metadata(self, text: str, source_file: str = "") -> List[str]:
        result = self.split_text_advanced(text)
        out = []
        for i, chunk in enumerate(result.chunks, 1):
            if result.total_chunks > 1:
                t = self.token_counter.count_tokens(chunk.content)
                label = f"[File: {source_file} - Chunk {i}/{result.total_chunks} ({t} tokens)]"
                out.append(f"{label}\n{chunk.content}")
            else:
                out.append(f"[File: {source_file}]\n{chunk.content}")
        return out

    def split_by_paragraphs(self, text: str) -> List[str]:
        if not text:
            return []
        paras = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
        if len(paras) < 2:
            paras = [p.strip() for p in text.split('\n') if p.strip()]
        return paras

    def _preprocess(self, text: str) -> str:
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def _split(self, text: str) -> List[TextChunk]:
        chunks, pos, cid = [], 0, 0
        while pos < len(text):
            end = self._find_end(text, pos)
            if end > pos:
                content = text[pos:end].strip()
                if content:
                    t = self.token_counter.count_tokens(content)
                    chunks.append(TextChunk(content, pos, end, t, cid))
                    cid += 1
            pos = end
        return chunks

    def _find_end(self, text: str, start: int) -> int:
        max_end = min(start + int(self.max_tokens * 4), len(text))
        if max_end >= len(text):
            return len(text)
        for pattern in self.split_patterns:
            matches = list(re.finditer(pattern, text[start:max_end]))
            if matches:
                target = int(self.max_tokens * 3)
                m = min(matches, key=lambda x: abs(x.end() - target))
                candidate = start + m.end()
                if self.token_counter.count_tokens(text[start:candidate]) <= self.max_tokens:
                    return candidate
        return max_end

    def _merge_small(self, chunks: List[TextChunk]) -> List[TextChunk]:
        if len(chunks) <= 1:
            return chunks
        min_tokens = max(100, self.max_tokens // 4)
        merged, i = [], 0
        while i < len(chunks):
            c = chunks[i]
            if (c.tokens < min_tokens and i + 1 < len(chunks) and
                    c.tokens + chunks[i+1].tokens <= self.max_tokens):
                nc = chunks[i+1]
                content = c.content + "\n\n" + nc.content
                merged.append(TextChunk(content, c.start_pos, nc.end_pos,
                                        self.token_counter.count_tokens(content), c.chunk_id))
                i += 2
            else:
                merged.append(c)
                i += 1
        return merged

    def _add_overlap(self, chunks: List[TextChunk]) -> List[TextChunk]:
        if len(chunks) <= 1:
            return chunks
        out = []
        for i, chunk in enumerate(chunks):
            content = chunk.content
            if i > 0:
                prev = self._overlap_text(chunks[i-1].content, True)
                if prev:
                    content = prev + "\n...\n" + content
            if i < len(chunks) - 1:
                nxt = self._overlap_text(chunks[i+1].content, False)
                if nxt:
                    content = content + "\n...\n" + nxt
            t = self.token_counter.count_tokens(content)
            out.append(TextChunk(content, chunk.start_pos, chunk.end_pos, t, chunk.chunk_id))
        return out

    def _overlap_text(self, text: str, from_end: bool) -> str:
        target = min(self.overlap_tokens, self.max_tokens // 4)
        words = text.split()
        if from_end:
            words = reversed(words)
        result, total = [], 0
        for w in words:
            wt = self.token_counter.count_tokens(w)
            if total + wt > target:
                break
            result.append(w)
            total += wt
        if from_end:
            result = list(reversed(result))
        return ' '.join(result)

    def get_stats(self, text: str) -> Dict:
        if not text:
            return {"chars": 0, "tokens": 0, "estimated_chunks": 0}
        tokens = self.token_counter.count_tokens(text)
        chunks = max(1, (tokens + self.max_tokens - 1) // self.max_tokens)
        return {"chars": len(text), "tokens": tokens, "estimated_chunks": chunks}
