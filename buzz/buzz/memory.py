"""Memória externa: SQLite + FTS5, com proveniência. Nada aqui é tratado como verdade."""
import re
import sqlite3
import threading
import time

SOURCES = ("user", "retrieved", "generated")
_REMEMBER = re.compile(r"^\s*(?:por favor[, ]*)?(?:lembre(?:-se)?(?: de)?|anote|guarde|salve|memorize|registre|grave|grava|"
                       r"não esqueça|toma nota|adicione à memória|fica sabendo|quero que você saiba)\s*(?:isso|aí)?\s*(?:que|:)?\s*",
                       re.I)


def is_explicit_remember(text: str) -> bool:
    """Escrever na memória exige comando explícito e texto curto — nunca só o palpite do classificador."""
    return len(text) <= 400 and bool(_REMEMBER.match(text)) and _REMEMBER.match(text).end() > 0


def _words(s: str):
    return {w for w in re.findall(r"\w+", s.lower()) if len(w) > 3}


def strip_remember(text: str) -> str:
    return _REMEMBER.sub("", text, count=1).strip() or text.strip()


class Memory:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS mem(id INTEGER PRIMARY KEY, ts REAL, source TEXT, text TEXT);"
            "CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(text, content='mem', content_rowid='id',"
            " tokenize='unicode61 remove_diacritics 2');")

    def add(self, text: str, source: str = "user") -> int:
        if source not in SOURCES:
            raise ValueError("source inválida")
        with self.lock, self.db:
            cur = self.db.execute("INSERT INTO mem(ts,source,text) VALUES(?,?,?)", (time.time(), source, text[:2000]))
            self.db.execute("INSERT INTO mem_fts(rowid,text) VALUES(?,?)", (cur.lastrowid, text[:2000]))
            return cur.lastrowid

    def search(self, query: str, k: int = 3, sources=("user", "retrieved")):
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        if not words:
            return []
        q = " OR ".join(f'"{w}"' for w in words)
        marks = ",".join("?" * len(sources))
        with self.lock:
            rows = self.db.execute(
                f"SELECT m.id,m.source,m.text,bm25(mem_fts) s FROM mem_fts JOIN mem m ON m.id=mem_fts.rowid "
                f"WHERE mem_fts MATCH ? AND m.source IN ({marks}) ORDER BY s LIMIT ?", (q, *sources, k)).fetchall()
        qw = _words(query)
        need = 1 if len(qw) <= 4 else max(2, int(0.3 * len(qw)))  # consultas longas exigem mais sobreposição
        return [{"id": r[0], "source": r[1], "text": r[2], "score": -r[3]} for r in rows
                if len(qw & _words(r[2])) >= need]
