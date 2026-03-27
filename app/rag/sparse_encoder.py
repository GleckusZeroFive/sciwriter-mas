"""
Sparse encoder for BM25 keyword search.

Generates sparse vectors (token_hash -> term_frequency) from text.
Works with Qdrant Modifier.IDF for server-side IDF weighting.

No external ML dependencies (except optional pymorphy3 for Russian lemmatization).
"""

import logging
import re
from collections import Counter

from qdrant_client.http.models import SparseVector

logger = logging.getLogger(__name__)

VOCAB_SIZE = 2**18

_STOP_WORDS_RU = frozenset({
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "её", "мне",
    "быть", "вот", "от", "меня", "ещё", "нет", "о", "из", "ему",
    "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже",
    "или", "ни", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего",
    "ей", "мочь", "они", "тут", "где", "есть", "надо", "ней",
    "для", "мы", "тебя", "их", "чем", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под",
    "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один",
    "почти", "мой", "тем", "чтобы", "нее", "сейчас",
    "куда", "зачем", "весь", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше",
    "тот", "через", "эти", "нас", "про", "всего", "них",
    "много", "разве", "три", "впрочем",
    "хорошо", "свой", "перед", "иногда", "лучше",
    "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "это", "эта",
})

_STOP_WORDS_EN = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "need", "dare", "ought", "used", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "up", "it", "its", "i", "me", "my", "he", "him",
    "his", "she", "her", "we", "us", "our", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
})

STOP_WORDS = _STOP_WORDS_RU | _STOP_WORDS_EN

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_MIN_TOKEN_LEN = 2

try:
    import pymorphy3 as _pymorphy3_module
    _morph = _pymorphy3_module.MorphAnalyzer()
    _PYMORPHY3_AVAILABLE = True
    logger.debug("pymorphy3 loaded, lemmatization active")
except ImportError:
    _PYMORPHY3_AVAILABLE = False
    logger.warning(
        "pymorphy3 not installed — BM25 runs without lemmatization. "
        "Install: pip install pymorphy3 pymorphy3-dicts-ru"
    )


def _lemmatize_ru(token: str) -> str:
    if not _PYMORPHY3_AVAILABLE:
        return token
    if not all("\u0400" <= c <= "\u04FF" for c in token):
        return token
    parsed = _morph.parse(token)
    if parsed:
        return parsed[0].normal_form
    return token


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    result = []
    for t in tokens:
        if len(t) < _MIN_TOKEN_LEN or t in STOP_WORDS:
            continue
        lemma = _lemmatize_ru(t)
        if lemma in STOP_WORDS or len(lemma) < _MIN_TOKEN_LEN:
            continue
        result.append(lemma)
    return result


def _token_to_index(token: str) -> int:
    h = 0x811C9DC5
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % VOCAB_SIZE


def encode_sparse(text: str) -> SparseVector:
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[0], values=[0.0])
    counts = Counter(_token_to_index(t) for t in tokens)
    indices = sorted(counts.keys())
    values = [float(counts[idx]) for idx in indices]
    return SparseVector(indices=indices, values=values)


def encode_sparse_query(text: str) -> SparseVector:
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[0], values=[0.0])
    unique_indices = sorted(set(_token_to_index(t) for t in tokens))
    values = [1.0] * len(unique_indices)
    return SparseVector(indices=unique_indices, values=values)
