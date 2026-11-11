"""Algoritmos de resumen extractivo.

Se implementan dos enfoques clásicos de PLN y una etapa de diversidad:

* **TF-IDF + MMR**: puntúa cada oración por su similitud con el centroide
  del documento y selecciona las más relevantes evitando redundancia.
* **TextRank**: construye un grafo de oraciones y aplica PageRank para
  encontrar las oraciones más "centrales".

Todo está implementado con la biblioteca estándar (sin scikit-learn ni
networkx) para que los algoritmos sean visibles y explicables en el código.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .preprocessing import Sentence

METHOD_LABELS = {
    "tfidf": "TF-IDF + MMR",
    "textrank": "TextRank",
}


@dataclass(frozen=True)
class SummaryResult:
    """Resultado completo de un resumen."""

    method: str
    summary: str
    sentences: tuple[Sentence, ...]
    selected: tuple[int, ...]
    scores: tuple[float, ...]
    config: dict[str, object] = field(default_factory=dict)

    @property
    def method_label(self) -> str:
        label = METHOD_LABELS.get(self.method, self.method)
        if self.method == "tfidf" and not self.config.get("use_mmr", True):
            return "TF-IDF"
        return label

    @property
    def selected_sentences(self) -> list[Sentence]:
        return [self.sentences[i] for i in sorted(self.selected)]


def _l2_normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 0.0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def build_tfidf(sentences: list[Sentence]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Calcula vectores TF-IDF normalizados y el IDF de cada término.

    TF aumentado: ``0.5 + 0.5 * tf / max_tf`` para no favorecer en exceso a
    las oraciones largas. IDF suavizado: ``log((N + 1) / (df + 1)) + 1``.
    """

    n = len(sentences)
    if n == 0:
        return [], {}

    document_frequency: Counter[str] = Counter()
    for sentence in sentences:
        document_frequency.update(set(sentence.stems))

    idf = {
        term: math.log((n + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }

    vectors: list[dict[str, float]] = []
    for sentence in sentences:
        counts = Counter(sentence.stems)
        max_tf = max(counts.values()) if counts else 1
        vector = {
            term: (0.5 + 0.5 * count / max_tf) * idf.get(term, 1.0)
            for term, count in counts.items()
        }
        vectors.append(_l2_normalize(vector))

    return vectors, idf


def build_similarity_matrix(vectors: list[dict[str, float]]) -> list[dict[int, float]]:
    """Matriz dispersa de similitud coseno entre oraciones.

    En lugar de una matriz densa O(n²), se usa un índice invertido: solo se
    calculan los pares que comparten al menos un término. Cada fila es un
    diccionario ``{índice_de_otra_oración: similitud}``.
    """

    n = len(vectors)
    similarities: list[dict[int, float]] = [dict() for _ in range(n)]

    inverted_index: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, vector in enumerate(vectors):
        for term, weight in vector.items():
            inverted_index[term].append((index, weight))

    for postings in inverted_index.values():
        size = len(postings)
        if size < 2:
            continue
        for i in range(size):
            row_i, weight_i = postings[i]
            for j in range(i + 1, size):
                row_j, weight_j = postings[j]
                value = weight_i * weight_j
                similarities[row_i][row_j] = similarities[row_i].get(row_j, 0.0) + value
                similarities[row_j][row_i] = similarities[row_j].get(row_i, 0.0) + value

    return similarities


def _cosine_with_centroid(vectors: list[dict[str, float]]) -> list[float]:
    centroid: dict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, weight in vector.items():
            centroid[term] += weight
    centroid_normalized = _l2_normalize(dict(centroid))

    scores: list[float] = []
    for vector in vectors:
        score = sum(weight * centroid_normalized.get(term, 0.0) for term, weight in vector.items())
        scores.append(score)
    return scores


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def pagerank(
    similarities: list[dict[int, float]],
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> list[float]:
    """PageRank sobre un grafo ponderado no dirigido de oraciones.

    ``score(i) = (1-d)/N + d * Σ_j (w_ji / Σ_k w_jk) * score(j)``

    Los nodos sin aristas (oraciones sin vocabulario en común con el resto)
    se tratan como "dangling nodes" y su masa se reparte uniformemente.
    """

    n = len(similarities)
    if n == 0:
        return []

    out_weight = [sum(row.values()) for row in similarities]
    scores = [1.0 / n] * n

    for _ in range(max_iterations):
        new_scores = [(1.0 - damping) / n] * n
        dangling_mass = 0.0

        for i in range(n):
            if out_weight[i] <= 0.0:
                dangling_mass += scores[i]
                continue
            factor = damping * scores[i] / out_weight[i]
            for j, weight in similarities[i].items():
                new_scores[j] += factor * weight

        if dangling_mass > 0.0:
            share = damping * dangling_mass / n
            for j in range(n):
                new_scores[j] += share

        difference = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if difference < tolerance:
            break

    total = sum(scores)
    if total <= 0.0:
        return [1.0 / n] * n
    return [score / total for score in scores]


def mmr_select(
    relevance: list[float],
    similarities: list[dict[int, float]],
    k: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """Selección Maximal Marginal Relevance.

    ``MMR(i) = λ * relevancia(i) - (1-λ) * max_similitud(i, seleccionadas)``

    Con λ cercano a 1 prioriza relevancia; con λ cercano a 0 prioriza
    diversidad (evita repetir la misma idea con otras palabras).
    """

    n = len(relevance)
    k = max(0, min(k, n))
    selected: list[int] = []
    remaining = set(range(n))
    max_similarity = [0.0] * n

    while len(selected) < k and remaining:
        best_index = -1
        best_score = -math.inf

        for i in remaining:
            score = lambda_param * relevance[i] - (1.0 - lambda_param) * max_similarity[i]
            if score > best_score:
                best_score = score
                best_index = i

        selected.append(best_index)
        remaining.remove(best_index)

        for i in remaining:
            similarity = similarities[best_index].get(i, 0.0)
            if similarity > max_similarity[i]:
                max_similarity[i] = similarity

    return selected


def _evenly_sample(sentences: list[Sentence], limit: int) -> list[Sentence]:
    """Reduce documentos enormes manteniendo oraciones de todo el texto."""

    if len(sentences) <= limit:
        return sentences
    step = len(sentences) / limit
    return [sentences[int(i * step)] for i in range(limit)]


def summarize(
    sentences: list[Sentence],
    method: str = "tfidf",
    ratio: float = 0.2,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    position_weight: float = 0.1,
    max_sentences: int = 2000,
) -> SummaryResult:
    """Genera un resumen extractivo.

    Parámetros
    ----------
    sentences:
        Oraciones preprocesadas (ver :func:`src.preprocessing.preprocess_text`).
    method:
        ``"tfidf"`` o ``"textrank"``.
    ratio:
        Proporción de oraciones que se conservarán (0 < ratio <= 1).
    use_mmr:
        Activa la selección con diversidad.
    mmr_lambda:
        Balance relevancia/diversidad de MMR (0 a 1).
    position_weight:
        Peso de la posición inicial/final en el método TF-IDF. TextRank no
        usa este sesgo para mantener su formulación original.
    max_sentences:
        Límite de oraciones para acotar el coste de la matriz de similitud.
    """

    if not sentences:
        raise ValueError("No hay oraciones para resumir.")
    if method not in METHOD_LABELS:
        raise ValueError(f"Método desconocido: {method}")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("La proporción debe estar entre 0 y 1.")

    original_count = len(sentences)
    truncated = original_count > max_sentences
    working_sentences = _evenly_sample(sentences, max_sentences)

    vectors, _idf = build_tfidf(working_sentences)
    similarities = build_similarity_matrix(vectors)

    tfidf_relevance = _cosine_with_centroid(vectors)
    if position_weight > 0.0:
        n = len(working_sentences)
        first_block = max(1, int(0.15 * n))
        last_block_start = int(0.90 * n)
        position_prior = [
            1.0 if i < first_block else (0.5 if i >= last_block_start else 0.0)
            for i in range(n)
        ]
        tfidf_relevance = [
            (1.0 - position_weight) * relevance + position_weight * prior
            for relevance, prior in zip(tfidf_relevance, position_prior)
        ]
    tfidf_relevance = _minmax(tfidf_relevance)

    has_edges = any(row for row in similarities)

    if method == "tfidf":
        relevance = tfidf_relevance
        scores = relevance
    else:
        if has_edges:
            scores = _minmax(pagerank(similarities))
        else:  # Sin vocabulario compartido, PageRank no aporta información.
            scores = tfidf_relevance
        relevance = scores

    n = len(working_sentences)
    k = max(1, min(n, round(n * ratio)))

    if use_mmr and k < n:
        selected = mmr_select(relevance, similarities, k, mmr_lambda)
    else:
        selected = sorted(range(n), key=lambda i: relevance[i], reverse=True)[:k]

    selected = sorted(selected)
    summary = " ".join(working_sentences[i].text for i in selected)

    return SummaryResult(
        method=method,
        summary=summary,
        sentences=tuple(working_sentences),
        selected=tuple(selected),
        scores=tuple(round(score, 6) for score in scores),
        config={
            "ratio": ratio,
            "use_mmr": use_mmr,
            "mmr_lambda": mmr_lambda,
            "position_weight": position_weight if method == "tfidf" else 0.0,
            "sentence_count": n,
            "original_sentence_count": original_count,
            "truncated": truncated,
            "fallback_to_tfidf": method == "textrank" and not has_edges,
        },
    )


def compare_summaries(first: SummaryResult, second: SummaryResult) -> dict[str, object]:
    """Compara dos resúmenes con el índice de Jaccard sobre sus oraciones."""

    first_set = set(first.selected)
    second_set = set(second.selected)
    intersection = first_set & second_set
    union = first_set | second_set
    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "jaccard": jaccard,
        "common": len(intersection),
        "union": len(union),
        "first_only": len(first_set - second_set),
        "second_only": len(second_set - first_set),
    }
