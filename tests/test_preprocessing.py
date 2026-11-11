from __future__ import annotations

from src.preprocessing import (
    detect_language,
    normalize_token,
    preprocess_text,
    split_sentences,
    strip_accents,
)


def test_split_sentences_protects_abbreviations():
    text = "El Dr. Pérez llegó a las 10:30 a. m. El precio fue de 3.14 dólares."
    assert split_sentences(text) == [
        "El Dr. Pérez llegó a las 10:30 a. m.",
        "El precio fue de 3.14 dólares.",
    ]


def test_split_sentences_soft_abbreviations():
    text = "Se usaron varias técnicas, etc. El resultado fue positivo. Ver el No. 5."
    assert split_sentences(text) == [
        "Se usaron varias técnicas, etc.",
        "El resultado fue positivo.",
        "Ver el No. 5.",
    ]


def test_split_sentences_handles_quotes_and_newlines():
    text = 'Dijo: "Hola." Luego se fue.\n\nNueva idea.'
    sentences = split_sentences(text)
    assert sentences[0] == 'Dijo: "Hola."'
    assert "Luego se fue." in sentences
    assert sentences[-1] == "Nueva idea."


def test_strip_accents_keeps_enye():
    assert strip_accents("mañana") == "mañana"
    assert strip_accents("canción") == "cancion"
    assert normalize_token("MÁS") == "mas"


def test_preprocess_removes_stopwords_and_duplicates():
    text = (
        "La inteligencia artificial permite analizar grandes documentos. "
        "La inteligencia artificial permite analizar grandes documentos. "
        "Los modelos de lenguaje procesan texto."
    )
    sentences = preprocess_text(text, language="spanish", min_words=3)
    assert len(sentences) == 2
    assert "inteligencia" in sentences[0].tokens
    assert "la" not in sentences[0].tokens
    assert sentences[0].stems == tuple(
        stem for stem in sentences[0].stems
    )
    assert len(sentences[0].stems) == len(sentences[0].surfaces)


def test_preprocess_filters_short_sentences():
    text = "Título. Esto es una oración suficientemente larga para el resumen."
    sentences = preprocess_text(text, language="spanish", min_words=6)
    assert len(sentences) == 1
    assert "suficientemente" in sentences[0].tokens


def test_detect_language():
    spanish = "El perro corre por el parque y juega con la pelota."
    english = "The quick brown fox jumps over the lazy dog and runs away."
    assert detect_language(spanish) == "spanish"
    assert detect_language(english) == "english"
