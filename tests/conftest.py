from __future__ import annotations

import pytest

from src.preprocessing import preprocess_text

TEXT = (
    "La inteligencia artificial es una rama de la informática que estudia "
    "cómo las máquinas pueden aprender. "
    "El procesamiento de lenguaje natural permite analizar y comprender texto "
    "escrito por humanos. "
    "Los modelos de lenguaje aprenden patrones estadísticos a partir de grandes "
    "volúmenes de datos. "
    "La extracción de información identifica las ideas principales y las "
    "entidades relevantes de un documento. "
    "Los resumidores automáticos reducen la longitud de un texto conservando su "
    "contenido esencial. "
    "La evaluación de un resumen mide la cobertura de las ideas y la coherencia "
    "del resultado. "
    "El análisis de frecuencia de términos ayuda a detectar los conceptos más "
    "importantes. "
    "La similitud coseno compara dos vectores y produce un valor entre cero y uno."
)


@pytest.fixture
def sentences():
    return preprocess_text(TEXT, language="spanish", min_words=4)
