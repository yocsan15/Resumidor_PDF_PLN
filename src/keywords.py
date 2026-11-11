"""Extracción de palabras clave mediante TF-IDF global.

La idea es sencilla: un término es importante si aparece muchas veces en el
documento (TF) pero no en todas las oraciones (IDF). Se agrupan variantes
con el stem y se muestra la forma superficial más frecuente.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .preprocessing import Sentence


@dataclass(frozen=True)
class Keyword:
    """Palabra clave con su puntuación normalizada y frecuencia."""

    term: str
    score: float
    frequency: int
    stem: str = ""


def extract_keywords(sentences: list[Sentence], top_n: int = 15) -> list[Keyword]:
    """Devuelve las ``top_n`` palabras clave más relevantes del documento."""

    if not sentences or top_n <= 0:
        return []

    n = len(sentences)
    document_frequency: Counter[str] = Counter()
    for sentence in sentences:
        document_frequency.update(set(sentence.stems))

    idf = {
        term: math.log((n + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }

    frequency: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)

    for sentence in sentences:
        for stem, surface in zip(sentence.stems, sentence.surfaces):
            frequency[stem] += 1
            surfaces[stem][surface] += 1

    scored: list[Keyword] = []
    for stem, count in frequency.items():
        score = count * idf.get(stem, 1.0)
        surface = surfaces[stem].most_common(1)[0][0] if surfaces[stem] else stem
        scored.append(
            Keyword(
                term=surface,
                score=score,
                frequency=count,
                stem=stem,
            )
        )

    scored.sort(key=lambda keyword: (-keyword.score, keyword.term))
    scored = scored[:top_n]

    max_score = scored[0].score if scored else 1.0
    if max_score <= 0.0:
        max_score = 1.0

    return [
        Keyword(
            term=keyword.term,
            score=round(keyword.score / max_score, 4),
            frequency=keyword.frequency,
            stem=keyword.stem,
        )
        for keyword in scored
    ]
