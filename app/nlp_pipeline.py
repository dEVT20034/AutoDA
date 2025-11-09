"""
Utility functions for the NLP Automation workflow.

These helpers keep the text-processing logic isolated so the Flask routes can
focus on orchestration and persistence.
"""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


SUPPORTED_TEXT_EXTENSIONS = {".csv", ".txt", ".zip"}
NORMALIZE_PATTERN = r"[^a-z0-9\s]"
WHITESPACE_PATTERN = r"\s+"


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if missing."""
    resources = ["stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:  # pragma: no cover - defensive download
            nltk.download(resource)


def load_text_dataset(path: Path, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load text data from CSV, TXT, or ZIP archive into a DataFrame."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        metadata = {
            "source_type": "csv",
            "rows": len(df),
            "columns": df.columns.tolist(),
        }
        return df, metadata

    if suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = _split_txt_documents(raw_text)
        df = pd.DataFrame({"text": chunks})
        metadata = {
            "source_type": "txt",
            "rows": len(df),
            "columns": ["text"],
        }
        return df, metadata

    if suffix == ".zip":
        documents: list[str] = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.lower().endswith(".txt"):
                    with archive.open(member) as handle:
                        text_bytes = handle.read()
                        documents.append(
                            text_bytes.decode("utf-8", errors="ignore").strip()
                        )
        df = pd.DataFrame({"text": [doc for doc in documents if doc]})
        metadata = {
            "source_type": "zip",
            "rows": len(df),
            "columns": ["text"],
        }
        return df, metadata

    raise ValueError(f"Unsupported file extension for NLP upload: {suffix}")


def _split_txt_documents(raw_text: str) -> List[str]:
    """Split a raw TXT payload into discrete documents."""
    if not raw_text.strip():
        return []
    if "\n\n" in raw_text:
        chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
        if chunks:
            return chunks
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines or [raw_text.strip()]


def preprocess_texts(
    texts: pd.Series,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> Tuple[pd.Series, List[Dict[str, str]]]:
    """Normalize, tokenize, and optionally lemmatize text data."""
    ensure_nltk_resources()
    stopword_set: set[str] = set()
    if remove_stopwords:
        stopword_set = set(stopwords.words("english"))
    lemmatizer: Optional[WordNetLemmatizer] = WordNetLemmatizer() if lemmatize else None

    normalized = texts.fillna("").astype(str).str.lower()
    normalized = normalized.str.replace(NORMALIZE_PATTERN, " ", regex=True)
    normalized = normalized.str.replace(WHITESPACE_PATTERN, " ", regex=True).str.strip()

    if remove_stopwords or lemmatize:
        def _transform(doc: str) -> str:
            tokens = doc.split()
            if remove_stopwords and stopword_set:
                tokens = [token for token in tokens if token not in stopword_set]
            if lemmatizer:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return " ".join(tokens)

        cleaned_series = normalized.apply(_transform)
    else:
        cleaned_series = normalized

    samples: list[Dict[str, str]] = []
    for original, cleaned in zip(texts.fillna("").astype(str), cleaned_series):
        if len(samples) >= 5:
            break
        samples.append(
            {
                "original": original[:500],
                "cleaned": cleaned[:500],
            }
        )

    return cleaned_series.astype("string"), samples


def profile_texts(
    cleaned_texts: pd.Series,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Produce lightweight EDA style metrics for text corpora."""
    series = cleaned_texts.fillna("").astype(str)
    total_documents = int(len(series))
    empty_count = int((series.str.strip() == "").sum())

    if sample_size and sample_size < total_documents:
        sample = series.sample(sample_size, random_state=42)
    else:
        sample = series

    token_lists = sample.apply(lambda text: [tok for tok in text.split() if tok])
    word_counts = token_lists.apply(len)

    vocabulary = set(token for tokens in token_lists for token in tokens)
    word_counter = Counter(token for tokens in token_lists for token in tokens)
    bigram_counter = Counter(
        " ".join(pair)
        for tokens in token_lists
        for pair in zip(tokens, tokens[1:])
        if len(pair) == 2
    )

    def _top(counter: Counter, limit: int = 10) -> List[dict[str, Any]]:
        return [{"term": term, "count": int(count)} for term, count in counter.most_common(limit)]

    length_distribution = _length_distribution(word_counts)

    return {
        "documents": total_documents,
        "empty": empty_count,
        "average_word_count": float(word_counts.mean()) if not word_counts.empty else 0.0,
        "median_word_count": float(word_counts.median()) if not word_counts.empty else 0.0,
        "vocabulary_size": len(vocabulary),
        "top_words": _top(word_counter),
        "top_bigrams": _top(bigram_counter),
        "length_distribution": length_distribution,
        "sampled_documents": int(len(sample)),
    }


def _length_distribution(word_counts: pd.Series) -> List[dict[str, Any]]:
    """Bucket documents into short / medium / long groups."""
    if word_counts.empty:
        return [
            {"bucket": "Short (≤20 words)", "count": 0},
            {"bucket": "Medium (21-100 words)", "count": 0},
            {"bucket": "Long (>100 words)", "count": 0},
        ]

    short = int((word_counts <= 20).sum())
    medium = int(((word_counts > 20) & (word_counts <= 100)).sum())
    long = int((word_counts > 100).sum())
    return [
        {"bucket": "Short (≤20 words)", "count": short},
        {"bucket": "Medium (21-100 words)", "count": medium},
        {"bucket": "Long (>100 words)", "count": long},
    ]


def extract_tfidf_features(
    cleaned_texts: pd.Series,
    max_features: int = 5000,
) -> Tuple[TfidfVectorizer, sparse.csr_matrix, Dict[str, Any]]:
    """Vectorize cleaned text with TF-IDF (unigrams + bigrams)."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(cleaned_texts.fillna("").astype(str))
    feature_count = len(vectorizer.get_feature_names_out())
    summary = {
        "documents": matrix.shape[0],
        "feature_count": feature_count,
        "max_features": max_features,
        "sparsity": 1.0 - (matrix.count_nonzero() / (matrix.shape[0] * feature_count))
        if matrix.shape[0] and feature_count
        else 1.0,
    }
    return vectorizer, matrix, summary


def train_quick_text_classifier(
    vectorizer: TfidfVectorizer,
    cleaned_texts: pd.Series,
    labels: pd.Series,
    test_size: float = 0.2,
) -> Tuple[Optional[Dict[str, Any]], Optional[LogisticRegression]]:
    """Train a lightweight classifier when labels are provided."""
    if labels is None:
        return None, None

    label_series = labels.replace("", np.nan).dropna()
    mask = labels.index.isin(label_series.index)
    if label_series.nunique() < 2 or label_series.empty:
        return (
            {
                "status": "insufficient_labels",
                "message": "Need at least two distinct label values to train a classifier.",
            },
            None,
        )

    cleaned_subset = cleaned_texts.loc[mask]
    X_all = vectorizer.transform(cleaned_subset.fillna(""))
    y_all = label_series.astype(str)

    stratify = y_all if y_all.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    if X_train.shape[0] < 2 or X_test.shape[0] < 1:
        return (
            {
                "status": "insufficient_samples",
                "message": "Not enough labeled examples after the split to train/test a classifier.",
            },
            None,
        )

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=200,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics_summary = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }

    labels_order = list(model.classes_)
    cm = confusion_matrix(y_test, predictions, labels=labels_order)
    metrics_summary["confusion_matrix"] = {
        "labels": labels_order,
        "matrix": cm.tolist(),
    }
    metrics_summary["status"] = "trained"

    return metrics_summary, model


__all__ = [
    "SUPPORTED_TEXT_EXTENSIONS",
    "ensure_nltk_resources",
    "load_text_dataset",
    "preprocess_texts",
    "profile_texts",
    "extract_tfidf_features",
    "train_quick_text_classifier",
]
