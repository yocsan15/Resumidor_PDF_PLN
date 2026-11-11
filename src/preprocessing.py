"""Limpieza, segmentación y tokenización de texto para el resumidor.

El módulo implementa el preprocesamiento clásico de PLN:

1. Normalización Unicode y de espacios.
2. Protección de abreviaturas, decimales e iniciales para no cortar
   oraciones por error.
3. Segmentación en oraciones.
4. Tokenización con expresiones regulares.
5. Eliminación de stopwords.
6. Stemming (Snowball) para agrupar variantes de una misma palabra.
7. Detección heurística del idioma (español / inglés).

No se descargan corpus de NLTK: el stemmer Snowball viene incluido en el
paquete y las stopwords se leen de ``resources/``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # NLTK es opcional: si no está, el sistema sigue funcionando sin stemming.
    from nltk.stem.snowball import SnowballStemmer

    _NLTK_AVAILABLE = True
except Exception:  # pragma: no cover - depende del entorno
    SnowballStemmer = None  # type: ignore[assignment]
    _NLTK_AVAILABLE = False


RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"

# Carácter privado de Unicode que usamos como "punto protegido".
_DOT = "\uE000"

# Abreviaturas que casi siempre van seguidas de un nombre o texto, por lo
# que su punto nunca cierra una oración.
_HARD_ABBREVIATIONS = (
    "sr",
    "sra",
    "srta",
    "dr",
    "dra",
    "ing",
    "lic",
    "mtro",
    "mtra",
    "prof",
    "profa",
    "inc",
    "ltd",
    "co",
    "corp",
    "dept",
    "univ",
    "av",
    "ud",
    "uds",
    "vd",
    "vds",
    "gral",
    "cnel",
    "tte",
    "sgte",
    "sig",
)

# Abreviaturas que pueden cerrar una oración ("... etc. El siguiente...").
# Solo se protegen si las sigue un dígito o una palabra en minúscula.
_SOFT_ABBREVIATIONS = (
    "etc",
    "pag",
    "pág",
    "pags",
    "págs",
    "fig",
    "figs",
    "num",
    "núm",
    "art",
    "arts",
    "cap",
    "caps",
    "sec",
    "secs",
    "vol",
    "vols",
    "ed",
    "eds",
    "ej",
    "aprox",
    "vs",
    "tel",
    "cel",
    "max",
    "máx",
    "min",
    "mín",
    "no",
    "nro",
    "nra",
    "et",
)

# Abreviaturas con varios puntos: se protegen antes que las de una palabra.
_MULTI_DOT_ABBREVIATIONS = (
    r"EE\.\s?UU\.",
    r"U\.\s?S\.\s?A\.",
    r"p\.\s?ej\.",
    r"a\.\s?C\.",
    r"d\.\s?C\.",
    r"i\.\s?e\.",
    r"e\.\s?g\.",
    r"a\.\s?m\.",
    r"p\.\s?m\.",
    r"et\s?al\.",
)

_HARD_ABBREV_RE = re.compile(
    r"\b(" + "|".join(sorted(_HARD_ABBREVIATIONS, key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)

_SOFT_ABBREV_RE = re.compile(
    r"\b((?i:"
    + "|".join(sorted(_SOFT_ABBREVIATIONS, key=len, reverse=True))
    + r"))\.(?=\s*(?:\d|[a-záéíóúüñ]))"
)

# Letras Unicode (incluye acentos y ñ). Excluye dígitos y guiones bajos.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Frontera de oración: signo de cierre + comillas opcionales + espacio,
# seguido de mayúscula, dígito o signo de apertura. También un salto de línea.
# El grupo de comillas se conserva y se reengancha a la oración anterior.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?…])([\"'»”’)]*)\s+(?=[\"'«¿¡(]*[A-ZÁÉÍÓÚÜÑ0-9])|\n+"
)


def normalize_whitespace(text: str) -> str:
    """Colapsa espacios, tabuladores y saltos de línea."""

    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    """Elimina tildes y diéresis, pero conserva la ñ.

    Se usa para comparar tokens y stopwords sin depender de la acentuación
    (``más`` y ``mas`` deben tratarse igual).
    """

    text = text.replace("ñ", "\u0001").replace("Ñ", "\u0001")
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_marks.replace("\u0001", "ñ")


def normalize_token(token: str) -> str:
    """Minúsculas + sin tildes, conservando la ñ."""

    return strip_accents(token.lower())


@lru_cache(maxsize=4)
def load_stopwords(language: str = "spanish") -> frozenset[str]:
    """Carga la lista de stopwords del idioma indicado.

    Los archivos viven en ``resources/`` y están normalizados en tiempo de
    carga, por lo que pueden escribirse con o sin tildes.
    """

    filename = {
        "spanish": "stopwords_es.txt",
        "english": "stopwords_en.txt",
    }.get(language, "stopwords_es.txt")

    path = RESOURCES_DIR / filename
    if not path.exists():
        return frozenset()

    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(normalize_token(line))
    return frozenset(words)


@lru_cache(maxsize=4)
def _get_stemmer(language: str):
    """Devuelve un stemmer Snowball o ``None`` si NLTK no está disponible."""

    if not _NLTK_AVAILABLE:
        return None
    name = "spanish" if language == "spanish" else "english"
    try:
        return SnowballStemmer(name)
    except Exception:  # pragma: no cover
        return None


def _protect_all_but_last_dot(value: str) -> str:
    """Protege los puntos internos y deja libre el punto final.

    Se usa con abreviaturas horarias como ``a. m.``: el punto final también
    puede cerrar la oración ("... a. m. El precio...").
    """

    last_dot = value.rfind(".")
    if last_dot < 0:
        return value
    return value[:last_dot].replace(".", _DOT) + value[last_dot:]


def _protect_punctuation(text: str) -> str:
    """Sustituye puntos conflictivos por un marcador temporal."""

    time_abbreviations = {r"a\.\s?m\.", r"p\.\s?m\."}

    for pattern in _MULTI_DOT_ABBREVIATIONS:
        if pattern in time_abbreviations:
            text = re.sub(
                pattern,
                lambda match: _protect_all_but_last_dot(match.group(0)),
                text,
                flags=re.IGNORECASE,
            )
        else:
            text = re.sub(
                pattern,
                lambda match: match.group(0).replace(".", _DOT),
                text,
                flags=re.IGNORECASE,
            )

    text = _HARD_ABBREV_RE.sub(lambda match: match.group(1) + _DOT, text)
    text = _SOFT_ABBREV_RE.sub(lambda match: match.group(1) + _DOT, text)

    # Iniciales: "J. Pérez", "M. Curie".
    text = re.sub(r"\b([A-ZÁÉÍÓÚÜÑ])\.", lambda match: match.group(1) + _DOT, text)

    # Decimales y separadores numéricos: 3.14, 1.000.
    text = re.sub(r"(\d)\.(\d)", lambda match: match.group(1) + _DOT + match.group(2), text)

    return text


def split_sentences(text: str) -> list[str]:
    """Segmenta un texto en oraciones.

    Es un segmentador basado en reglas, pensado para texto extraído de PDFs.
    Protege abreviaturas y decimales antes de buscar signos de cierre, y
    también considera los saltos de línea como fronteras (útil para listas).
    """

    if not text or not text.strip():
        return []

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Une palabras cortadas por guion al final de una línea.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = _protect_punctuation(text)

    parts = _SENTENCE_SPLIT_RE.split(text)

    # ``re.split`` con grupo de captura devuelve las comillas de cierre como
    # elementos separados. Las reenganchamos a la oración anterior.
    joined_parts: list[str] = []
    for part in parts:
        if part is None:
            continue
        if part and all(char in "\"'»”’)]" for char in part):
            if joined_parts:
                joined_parts[-1] += part
            continue
        joined_parts.append(part)

    sentences: list[str] = []
    for part in joined_parts:
        sentence = part.replace(_DOT, ".").strip()

        # Quita viñetas y numeración de listas.
        sentence = re.sub(r"^[\-–—•*·]\s*", "", sentence)
        sentence = re.sub(r"^\d{1,3}[.)]\s+", "", sentence)
        sentence = normalize_whitespace(sentence)

        if len(sentence) < 2:
            continue
        if not _TOKEN_RE.search(sentence):
            continue
        sentences.append(sentence)

    return sentences


def _looks_like_heading(sentence: str, words: list[str]) -> bool:
    """Detecta títulos y encabezados que no aportan al resumen.

    Los encabezados suelen carecer de puntuación final y son cortos. Se
    descartan si están numerados, en mayúsculas o si la mayoría de sus
    palabras empieza con mayúscula (estilo de título).
    """

    stripped = sentence.strip()
    if not stripped:
        return False

    core = stripped.rstrip("\"'»”’)")
    if core and core[-1] in ".!?…":
        return False
    if len(words) > 15:
        return False

    if re.match(r"^\d+(?:\.\d+)*[.)]?\s+", stripped):
        return True
    if stripped.isupper():
        return True

    capitalized = sum(1 for word in words if word[:1].isupper())
    return len(words) <= 12 and capitalized >= max(2, len(words) * 0.5)


@dataclass(frozen=True, slots=True)
class Sentence:
    """Una oración ya preprocesada.

    ``tokens``, ``stems`` y ``surfaces`` están alineados: la posición *i*
    corresponde al mismo término en las tres tuplas. ``surfaces`` conserva
    la forma original (minúsculas y con tildes) para mostrarla en la interfaz.
    """

    index: int
    text: str
    words: tuple[str, ...]
    tokens: tuple[str, ...]
    stems: tuple[str, ...]
    surfaces: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.words)


def preprocess_text(
    text: str,
    language: str = "spanish",
    min_words: int = 6,
    min_tokens: int = 3,
    use_stemming: bool = True,
    deduplicate: bool = True,
) -> list[Sentence]:
    """Convierte texto plano en una lista de :class:`Sentence`.

    Parámetros
    ----------
    text:
        Texto completo del documento.
    language:
        ``"spanish"`` o ``"english"``.
    min_words:
        Descarta oraciones con menos palabras (encabezados, pies de página).
    min_tokens:
        Mínimo de términos de contenido tras quitar stopwords.
    use_stemming:
        Si es ``True`` agrupa variantes con SnowballStemmer.
    deduplicate:
        Elimina oraciones repetidas (frecuentes en encabezados de PDF).
    """

    stopwords = load_stopwords(language)
    stemmer = _get_stemmer(language) if use_stemming else None

    result: list[Sentence] = []
    seen: set[str] = set()

    for raw_sentence in split_sentences(text):
        words = _TOKEN_RE.findall(raw_sentence)
        if len(words) < min_words:
            continue
        if _looks_like_heading(raw_sentence, words):
            continue

        tokens: list[str] = []
        stems: list[str] = []
        surfaces: list[str] = []

        for word in words:
            token = normalize_token(word)
            if len(token) < 2 or token in stopwords:
                continue
            stem = stemmer.stem(token) if stemmer is not None else token
            tokens.append(token)
            stems.append(stem)
            surfaces.append(word.lower())

        if len(tokens) < min_tokens:
            continue

        key = " ".join(stems)
        if deduplicate and key in seen:
            continue
        seen.add(key)

        result.append(
            Sentence(
                index=len(result),
                text=raw_sentence,
                words=tuple(word.lower() for word in words),
                tokens=tuple(tokens),
                stems=tuple(stems),
                surfaces=tuple(surfaces),
            )
        )

    return result


def detect_language(text: str, sample_chars: int = 12_000) -> str:
    """Heurística de detección de idioma basada en stopwords y caracteres.

    Compara cuántas stopwords distintas de cada idioma aparecen en una
    muestra del texto. Es suficiente para elegir la lista de stopwords y el
    stemmer, no pretende ser un detector lingüístico profesional.
    """

    sample = text[:sample_chars]
    tokens = {normalize_token(token) for token in _TOKEN_RE.findall(sample)}
    if not tokens:
        return "spanish"

    spanish_hits = len(tokens & load_stopwords("spanish"))
    english_hits = len(tokens & load_stopwords("english"))

    # Señales tipográficas del español.
    spanish_hits += sample.count("ñ") + sample.count("¿") + sample.count("¡")

    return "spanish" if spanish_hits >= english_hits else "english"


def most_common_surface(sentences: list[Sentence]) -> dict[str, str]:
    """Devuelve, por cada stem, la forma superficial más frecuente."""

    surfaces: dict[str, Counter[str]] = {}
    for sentence in sentences:
        for stem, surface in zip(sentence.stems, sentence.surfaces):
            surfaces.setdefault(stem, Counter())[surface] += 1
    return {
        stem: counter.most_common(1)[0][0]
        for stem, counter in surfaces.items()
        if counter
    }
