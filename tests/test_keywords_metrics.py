from __future__ import annotations

from src.keywords import extract_keywords
from src.metrics import compression_ratio, compute_stats, count_syllables
from src.summarizers import summarize


def test_extract_keywords(sentences):
    keywords = extract_keywords(sentences, top_n=5)
    assert len(keywords) == 5
    assert all(0.0 <= keyword.score <= 1.0 for keyword in keywords)
    assert all(
        first.score >= second.score
        for first, second in zip(keywords, keywords[1:])
    )
    assert all(keyword.term for keyword in keywords)


def test_compute_stats():
    stats = compute_stats("Hola mundo. Esto es una prueba sencilla.")
    assert stats.words == 7
    assert stats.sentences == 2
    assert stats.unique_words == 7
    assert 0.0 <= stats.lexical_diversity <= 1.0
    assert 0.0 <= stats.fernandez_huerta <= 100.0
    assert stats.reading_minutes > 0


def test_compression_ratio(sentences):
    original = compute_stats(" ".join(sentence.text for sentence in sentences))
    result = summarize(sentences, method="tfidf", ratio=0.25)
    summary = compute_stats(result.summary)
    ratio = compression_ratio(original, summary)
    assert 0.0 <= ratio <= 1.0
    assert ratio > 0.0


def test_count_syllables_spanish():
    assert count_syllables("a") == 1
    assert count_syllables("queso") == 2
    assert count_syllables("guitarra") == 3
    assert count_syllables("murciélago") == 4
    assert count_syllables("123") == 0
