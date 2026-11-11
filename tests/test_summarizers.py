from __future__ import annotations

import math

import pytest

from src.preprocessing import preprocess_text
from src.summarizers import (
    build_tfidf,
    compare_summaries,
    pagerank,
    summarize,
)


def test_tfidf_vectors_are_normalized(sentences):
    vectors, idf = build_tfidf(sentences)
    assert len(vectors) == len(sentences)
    assert idf
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        assert norm == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["tfidf", "textrank"])
def test_summarize_returns_expected_number_of_sentences(sentences, method):
    result = summarize(sentences, method=method, ratio=0.5)
    assert result.method == method
    assert len(result.selected) == 4
    assert list(result.selected) == sorted(result.selected)
    assert len(set(result.selected)) == len(result.selected)
    assert result.summary
    assert len(result.scores) == len(result.sentences)


def test_summarize_without_mmr(sentences):
    result = summarize(sentences, method="tfidf", ratio=0.25, use_mmr=False)
    assert len(result.selected) == 2
    assert result.config["use_mmr"] is False


def test_textrank_falls_back_when_no_shared_vocabulary():
    text = (
        "Alfa beta gamma delta epsilon. "
        "Zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron."
    )
    sentences = preprocess_text(text, language="spanish", min_words=3)
    result = summarize(sentences, method="textrank", ratio=0.34)
    assert result.config["fallback_to_tfidf"] is True
    assert result.summary


def test_compare_summaries_with_itself(sentences):
    result = summarize(sentences, method="tfidf", ratio=0.25)
    comparison = compare_summaries(result, result)
    assert comparison["jaccard"] == 1.0
    assert comparison["common"] == len(result.selected)


def test_pagerank_distributes_probability():
    similarities = [{1: 1.0}, {0: 1.0}]
    scores = pagerank(similarities)
    assert sum(scores) == pytest.approx(1.0)
    assert scores[0] == pytest.approx(scores[1])


def test_summarize_rejects_invalid_input(sentences):
    with pytest.raises(ValueError):
        summarize([], method="tfidf")
    with pytest.raises(ValueError):
        summarize(sentences, method="desconocido")
    with pytest.raises(ValueError):
        summarize(sentences, ratio=0.0)
