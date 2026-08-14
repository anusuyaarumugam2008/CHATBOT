from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Embeddings are intentionally opt-in. The built-in lexical retriever works
# without Torch, torchvision, or sentence-transformers.
SentenceTransformer = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAQ_FILENAME = "college_faq.csv"
FAQ_PATH = PROJECT_ROOT / "data" / FAQ_FILENAME
MODEL_NAME = "all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.34

_model: Any = None
_faq_frame: pd.DataFrame | None = None
_embeddings: np.ndarray | None = None
_token_sets: list[set[str]] | None = None


def _resolve_faq_path() -> Path:
    configured = os.getenv("CAMPUSAI_FAQ_PATH", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        PROJECT_ROOT / "data" / FAQ_FILENAME,
        Path.cwd() / "data" / FAQ_FILENAME,
        Path.cwd() / FAQ_FILENAME,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    for folder in [PROJECT_ROOT / "data", Path.cwd() / "data", Path.cwd()]:
        if folder.is_dir():
            csv_files = sorted(folder.glob("*.csv"))
            preferred = [path for path in csv_files if "faq" in path.name.lower() or "college" in path.name.lower()]
            if preferred:
                return preferred[0].resolve()
            if len(csv_files) == 1:
                return csv_files[0].resolve()
    return candidates[0]


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", value.lower())
    stop_words = {"a", "an", "the", "is", "are", "am", "i", "how", "can", "do", "does", "what", "where", "when", "for", "to", "of", "my", "on", "in", "and", "or"}
    return {word for word in words if word not in stop_words and len(word) > 1}


def _load_faqs() -> pd.DataFrame:
    global _faq_frame, _token_sets
    if _faq_frame is not None:
        return _faq_frame

    path = _resolve_faq_path()
    if not path.exists():
        raise FileNotFoundError(f"FAQ CSV was not found at {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    required = {"question", "answer"}
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"FAQ CSV is missing required column(s): {', '.join(sorted(missing))}")

    rename_map = {original: name for name, original in normalized.items() if name in {"id", "question", "answer", "category", "source"}}
    frame = frame.rename(columns=rename_map).copy()
    for column in ("category", "source"):
        if column not in frame.columns:
            frame[column] = "Not specified"
    for column in ("question", "answer", "category", "source"):
        frame[column] = frame[column].fillna("").map(_clean)
    frame = frame[(frame["question"] != "") & (frame["answer"] != "")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("FAQ CSV contains no complete question-and-answer records")

    _faq_frame = frame
    _token_sets = [_tokens(question) for question in frame["question"]]
    return frame


def _load_embeddings(frame: pd.DataFrame) -> np.ndarray | None:
    global _model, _embeddings, SentenceTransformer
    if os.getenv("CAMPUSAI_USE_EMBEDDINGS", "0").lower() not in {"1", "true", "yes"}:
        return None
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as TransformerModel
            SentenceTransformer = TransformerModel
        except BaseException:
            return None
    if _embeddings is not None:
        return _embeddings
    try:
        _model = SentenceTransformer(MODEL_NAME)
        _embeddings = np.asarray(_model.encode(frame["question"].tolist(), normalize_embeddings=True, show_progress_bar=False))
        return _embeddings
    except Exception:
        _model = None
        _embeddings = None
        return None


def _lexical_scores(query: str, frame: pd.DataFrame) -> np.ndarray:
    query_tokens = _tokens(query)
    query_lower = query.lower()
    scores = []
    for index, question in enumerate(frame["question"]):
        question_tokens = _token_sets[index] if _token_sets else _tokens(question)
        category_tokens = _tokens(str(frame.iloc[index].get("category", "")))
        union = query_tokens | question_tokens
        overlap = len(query_tokens & question_tokens) / max(1, len(query_tokens))
        jaccard = len(query_tokens & question_tokens) / max(1, len(union))
        category_bonus = 0.20 if query_tokens & category_tokens else 0.0
        phrase_bonus = 0.18 if query_lower in question.lower() or question.lower() in query_lower else 0.0
        scores.append(min(1.0, overlap * 0.54 + jaccard * 0.22 + category_bonus + phrase_bonus))
    return np.asarray(scores, dtype=float)


def _similarity_scores(query: str, frame: pd.DataFrame) -> np.ndarray:
    embeddings = _load_embeddings(frame)
    if embeddings is not None and _model is not None:
        try:
            query_embedding = np.asarray(_model.encode([query], normalize_embeddings=True, show_progress_bar=False))[0]
            cosine_scores = embeddings @ query_embedding
            return np.clip((cosine_scores + 1.0) / 2.0, 0.0, 1.0)
        except Exception:
            pass
    return _lexical_scores(query, frame)


def get_answer(query: str) -> dict[str, Any]:
    """Return the best FAQ match using the dashboard's expected response contract."""
    question = _clean(query)
    if not question:
        return {"found": False, "answer": "", "category": "", "source": "", "confidence": 0.0}

    frame = _load_faqs()
    scores = _similarity_scores(question, frame)
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index])
    row = frame.iloc[best_index]
    found = confidence >= CONFIDENCE_THRESHOLD

    return {
        "found": found,
        "answer": _clean(row["answer"]) if found else "",
        "category": _clean(row["category"]),
        "source": _clean(row["source"]),
        "confidence": confidence,
        "matched_question": _clean(row["question"]),
    }


def reset_cache() -> None:
    """Clear loaded FAQ and embedding state after replacing the CSV."""
    global _faq_frame, _embeddings, _token_sets, _model
    _faq_frame = None
    _embeddings = None
    _token_sets = None
    _model = None
