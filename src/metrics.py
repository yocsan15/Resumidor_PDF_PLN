"""Métricas descriptivas y de legibilidad del documento y del resumen."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .preprocessing import normalize_token, split_sentences, strip_accents

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_VOWEL_GROUP_RE = re.compile(r"[aeiou]+")

# Velocidad de lectura aproximada para texto académico en español.
WORDS_PER_MINUTE = 200


@dataclass(frozen=True)
class TextStats:
    """Estadísticas de un texto."""

    characters: int
    words: int
    sentences: int
    unique_words: int
    lexical_diversity: float
    average_sentence_length: float
    syllables: int
    syllables_per_100_words: float
    fernandez_huerta: float
    readability_label: str
    reading_minutes: float


def count_syllables(word: str) -> int:
    """Estimación de sílabas en español.

    Cuenta grupos vocálicos y aplica dos reglas básicas (``qu`` y ``gu``
    seguidas de vocal). No sustituye a un silabificador lingüístico, pero es
    suficiente para una estimación de legibilidad.
    """

    word = strip_accents(word.lower())
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    word = word.replace("qu", "q")
    word = re.sub(r"gu([ei])", r"g\1", word)
    groups = _VOWEL_GROUP_RE.findall(word)
    return max(1, len(groups))


def readability_label(score: float) -> str:
    """Interpretación cualitativa del índice Fernández-Huerta."""

    if score >= 90:
        return "Muy fácil"
    if score >= 80:
        return "Fácil"
    if score >= 70:
        return "Bastante fácil"
    if score >= 60:
        return "Normal"
    if score >= 50:
        return "Bastante difícil"
    if score >= 30:
        return "Difícil"
    return "Muy difícil"


def compute_stats(text: str) -> TextStats:
    """Calcula conteos, diversidad léxica, tiempo de lectura y legibilidad."""

    words = _WORD_RE.findall(text)
    word_count = len(words)
    sentences = split_sentences(text)
    sentence_count = len(sentences)
    characters = len(text)

    if word_count == 0 or sentence_count == 0:
        return TextStats(
            characters=characters,
            words=word_count,
            sentences=sentence_count,
            unique_words=0,
            lexical_diversity=0.0,
            average_sentence_length=0.0,
            syllables=0,
            syllables_per_100_words=0.0,
            fernandez_huerta=0.0,
            readability_label="N/D",
            reading_minutes=0.0,
        )

    normalized_words = [normalize_token(word) for word in words]
    unique_words = len(set(normalized_words))
    lexical_diversity = unique_words / word_count
    average_sentence_length = word_count / sentence_count

    syllables = sum(count_syllables(word) for word in words)
    syllables_per_100_words = syllables / word_count * 100.0

    # Índice Fernández-Huerta:
    # 206.84 - 0.60 * (sílabas por 100 palabras) - 1.02 * (palabras por oración)
    fernandez_huerta = (
        206.84
        - 0.60 * syllables_per_100_words
        - 1.02 * average_sentence_length
    )
    fernandez_huerta = max(0.0, min(100.0, fernandez_huerta))

    return TextStats(
        characters=characters,
        words=word_count,
        sentences=sentence_count,
        unique_words=unique_words,
        lexical_diversity=round(lexical_diversity, 4),
        average_sentence_length=round(average_sentence_length, 2),
        syllables=syllables,
        syllables_per_100_words=round(syllables_per_100_words, 2),
        fernandez_huerta=round(fernandez_huerta, 1),
        readability_label=readability_label(fernandez_huerta),
        reading_minutes=round(word_count / WORDS_PER_MINUTE, 2),
    )


def compression_ratio(original: TextStats, summary: TextStats) -> float:
    """Porcentaje de reducción en palabras (0 a 1)."""

    if original.words <= 0:
        return 0.0
    ratio = 1.0 - summary.words / original.words
    return max(0.0, min(1.0, ratio))


def reading_time_saved(original: TextStats, summary: TextStats) -> float:
    """Minutos de lectura ahorrados (nunca negativo)."""

    return max(0.0, original.reading_minutes - summary.reading_minutes)
