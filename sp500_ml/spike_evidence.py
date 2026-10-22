"""Retain retrieved evidence snapshots in Convex, including unmatched documents."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json

from .artifacts import exclusive_file
from .paths import NEWS_DIR

_documents = ContextVar("spike_evidence_documents", default=None)


@contextmanager
def collect_documents():
    documents = []
    token = _documents.set(documents)
    try:
        yield documents
    finally:
        _documents.reset(token)


def remember_document(text, **metadata):
    documents = _documents.get()
    if documents is not None:
        documents.append({**metadata, "contentHash": hashlib.sha256(text.encode()).hexdigest(),
                          "text": text[:100000], "textTruncated": len(text) > 100000})


def load_evidence(ticker, event_date):
    path = NEWS_DIR / "spike_evidence" / event_date / f"{ticker}.json"
    return json.loads(path.read_text()) if path.exists() else {"headlines": [], "documents": []}


def save_evidence(ticker, event_date, headlines, documents, retrieved_at):
    if not headlines and not documents:
        return
    path = NEWS_DIR / "spike_evidence" / event_date / f"{ticker}.json"
    with exclusive_file(NEWS_DIR / f".{ticker}.{event_date}.spike-evidence.lock"):
        existing = load_evidence(ticker, event_date)
        news = existing.get("headlines", []) + [{**row, "retrievedAt": retrieved_at} for row in headlines]
        # Keep the first observation of each content version, even if a publisher revises it.
        retained_news = {}
        for row in news:
            identity = (row["url"], row["published_at_utc"], row.get("title"), row.get("excerpt"))
            retained_news.setdefault(identity, row)
        retained_docs = {}
        for row in existing.get("documents", []) + documents:
            retained_docs.setdefault((row["url"], row["contentHash"]), row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ticker": ticker, "eventDate": event_date, "schemaVersion": 1,
            "headlines": list(retained_news.values()), "documents": list(retained_docs.values())}))
