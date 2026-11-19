"""Tiny local knowledge base: loads .md/.txt files from /knowledge and does BM25 search."""
import math
import re
from collections import Counter

from . import config

_TOKEN = re.compile(r"\w+", re.UNICODE)
# If the whole KB is smaller than this, just give all of it to the LLM (best answers).
FULL_KB_LIMIT = 8000


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class KnowledgeBase:
    def __init__(self):
        self.reload()

    def reload(self):
        self.chunks: list[str] = []
        files = sorted(config.KNOWLEDGE_DIR.glob("**/*"))
        for f in files:
            if f.suffix.lower() not in (".md", ".txt") or not f.is_file():
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            self.chunks.extend(self._split(text, f.stem))
        self.full_text = "\n\n".join(self.chunks)
        self.doc_tokens = [_tokens(c) for c in self.chunks]
        self.avgdl = sum(map(len, self.doc_tokens)) / max(len(self.doc_tokens), 1)
        df = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(self.doc_tokens)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @staticmethod
    def _split(text: str, source: str, max_chars: int = 700) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, buf = [], ""
        for p in paras:
            if len(buf) + len(p) > max_chars and buf:
                chunks.append(f"[{source}] {buf}")
                buf = ""
            buf = f"{buf}\n{p}" if buf else p
        if buf:
            chunks.append(f"[{source}] {buf}")
        return chunks

    def search(self, query: str, k: int = 4) -> str:
        if not self.chunks:
            return "(knowledge base is empty)"
        if len(self.full_text) <= FULL_KB_LIMIT:
            return self.full_text
        q = _tokens(query)
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * dl / self.avgdl))
            scores.append((s, i))
        scores.sort(reverse=True)
        top = [self.chunks[i] for s, i in scores[:k] if s > 0]
        return "\n\n".join(top) if top else "(no matching information found)"


kb = KnowledgeBase()
