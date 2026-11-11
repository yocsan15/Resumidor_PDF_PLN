# Explicación técnica

## 1. Estructura del código

```text
Proyecto_Final/
├── app.py                  # Interfaz web (Streamlit)
├── cli.py                  # Interfaz de consola (argparse)
├── src/
│   ├── __init__.py
│   ├── errors.py           # Excepciones del dominio
│   ├── pdf_extractor.py    # PDF → texto limpio
│   ├── preprocessing.py    # Texto → oraciones y tokens
│   ├── summarizers.py      # TF-IDF, TextRank, PageRank y MMR
│   ├── keywords.py         # Extracción de palabras clave
│   ├── metrics.py          # Estadísticas y legibilidad
│   └── utils.py            # Reportes y formato
├── resources/
│   ├── stopwords_es.txt
│   ├── stopwords_en.txt
│   ├── ejemplo_articulo.txt
│   └── ejemplo_articulo.pdf
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_summarizers.py
│   └── test_keywords_metrics.py
└── docs/
```

La separación busca que el núcleo (`src/`) no dependa de Streamlit. Tanto
`app.py` como `cli.py` importan las mismas funciones, lo que garantiza que ambos
producen resultados equivalentes.

---

## 2. Módulos, clases y funciones principales

### 2.1. `src/errors.py`

Define una jerarquía de excepciones:

```text
PDFSummarizerError
├── UnsupportedFileError
├── PDFEncryptedError
├── PDFScannedError
├── PDFEmptyError
├── PDFTooLargeError
└── NotEnoughTextError
```

Esto permite que la interfaz distinga entre un PDF cifrado y uno escaneado, por
ejemplo, sin inspeccionar mensajes de error de terceros.

### 2.2. `src/pdf_extractor.py`

| Elemento | Función |
| --- | --- |
| `PDFDocument` | Dataclass con texto, número de páginas, texto por página, metadatos, advertencias y nombre. |
| `extract_text_from_pdf` | Punto de entrada: valida, descifra si es posible, recorre páginas y devuelve el documento. |
| `_clean_page_text` | Normaliza, une guiones, elimina URLs, números de página y reconstruye párrafos. |
| `_remove_repeated_lines` | Detecta encabezados/pies repetidos en al menos 50 % de las páginas. |

Flujo interno:

```text
bytes → PdfReader → validar cifrado → recorrer páginas
      → extract_text() → _clean_page_text()
      → _remove_repeated_lines() → unir páginas → PDFDocument
```

### 2.3. `src/preprocessing.py`

| Elemento | Función |
| --- | --- |
| `Sentence` | Dataclass inmutable con índice, texto, palabras, tokens, stems y formas superficiales. |
| `split_sentences` | Segmentador basado en reglas. |
| `preprocess_text` | Pipeline completo de limpieza, tokenización, stopwords y stemming. |
| `load_stopwords` | Carga y normaliza las stopwords desde `resources/`. |
| `detect_language` | Heurística de idioma por stopwords y caracteres. |
| `strip_accents` | Elimina tildes conservando la ñ. |
| `_looks_like_heading` | Descarta títulos y encabezados. |

La clase `Sentence` mantiene tres tuplas alineadas:

```python
tokens   = ("inteligencia", "artificial", "permite", "analizar")
stems    = ("intelig", "artifici", "permit", "analiz")
surfaces = ("inteligencia", "artificial", "permite", "analizar")
```

La alineación permite calcular TF-IDF con stems y mostrar palabras legibles en la
interfaz.

### 2.4. `src/summarizers.py`

| Elemento | Función |
| --- | --- |
| `SummaryResult` | Resultado completo: método, resumen, oraciones, selección, puntajes y configuración. |
| `build_tfidf` | Calcula IDF y vectores TF-IDF normalizados. |
| `build_similarity_matrix` | Matriz dispersa de similitud coseno mediante índice invertido. |
| `pagerank` | PageRank con damping y manejo de nodos sin aristas. |
| `mmr_select` | Selección con relevancia y diversidad. |
| `summarize` | Orquesta el método elegido y devuelve el resumen. |
| `compare_summaries` | Índice de Jaccard entre dos resúmenes. |

### 2.5. `src/keywords.py`

| Elemento | Función |
| --- | --- |
| `Keyword` | Término, puntaje normalizado, frecuencia y stem. |
| `extract_keywords` | Ranking TF-IDF global de términos. |

### 2.6. `src/metrics.py`

| Elemento | Función |
| --- | --- |
| `TextStats` | Estadísticas de un texto. |
| `compute_stats` | Calcula palabras, oraciones, diversidad, sílabas, legibilidad y tiempo. |
| `count_syllables` | Estimación de sílabas por grupos vocálicos. |
| `compression_ratio` | Reducción de palabras del resumen. |
| `reading_time_saved` | Minutos ahorrados. |

### 2.7. `src/utils.py`

| Elemento | Función |
| --- | --- |
| `format_summary` | Devuelve párrafo o viñetas. |
| `selected_sentence_rows` | Filas para la tabla de oraciones seleccionadas. |
| `build_markdown_report` | Informe Markdown descargable. |
| `build_json_report` | Informe JSON serializable. |
| `report_to_json` | Serialización con acentos legibles. |
| `safe_filename` | Nombre de archivo seguro. |

### 2.8. `app.py`

La aplicación web se organiza así:

1. Configuración de página y CSS.
2. Funciones con `@st.cache_data`:
   - `cached_extract`
   - `cached_detect_language`
   - `analyze_document`
3. Barra lateral con controles.
4. Carga y validación del PDF.
5. Extracción y análisis.
6. Cuatro pestañas de resultados.

`@st.cache_data` evita repetir la extracción y el análisis cuando el usuario
cambia de pestaña o interactúa con un control que no modifica las entradas.

### 2.9. `cli.py`

Usa `argparse` y devuelve:

- `0` si todo sale bien;
- `2` si hay un error de validación o de PDF.

La salida principal va a `stdout`; los mensajes de estado a `stderr`.

---

## 3. Algoritmos implementados

### 3.1. TF-IDF

Implementación en `build_tfidf`:

```python
document_frequency.update(set(sentence.stems))
idf = log((n + 1) / (frequency + 1)) + 1
vector[term] = (0.5 + 0.5 * count / max_tf) * idf[term]
vector = vector / ||vector||
```

El uso de `set(sentence.stems)` es importante: el DF cuenta oraciones, no
apariciones. Si una palabra aparece tres veces en la misma oración, cuenta una
sola vez para el IDF.

### 3.2. Similitud coseno dispersa

En lugar de recorrer todos los pares de oraciones, se construye un índice
invertido:

```text
término → [(oración_1, peso_1), (oración_2, peso_2), ...]
```

Para cada término se multiplican los pesos de los pares de oraciones que lo
comparten y se acumulan en la matriz. Así se evitan los pares con similitud cero.

### 3.3. PageRank

La implementación sigue la formulación clásica:

```text
nuevo[i] = (1 - d) / N
para cada arista (i, j):
    nuevo[j] += d * score[i] * w_ij / suma_pesos[i]
```

Los nodos sin aristas se manejan como *dangling nodes*: su masa se reparte
uniformemente para que la suma de puntajes se mantenga en 1. La iteración se
detiene cuando la diferencia L1 es menor que `1e-6` o al llegar a 100
iteraciones.

### 3.4. MMR

La primera iteración elige la oración más relevante. Después, cada candidata se
penaliza por su similitud máxima con las ya seleccionadas. El parámetro λ controla
el equilibrio:

- `λ = 1`: solo relevancia (puede repetir ideas).
- `λ = 0.7`: valor por defecto, equilibrio razonable.
- `λ = 0`: solo diversidad (puede elegir oraciones poco relevantes).

### 3.5. Selección y reordenamiento

`summarize` calcula `k = round(N * ratio)`, con al menos una oración. Las
oraciones se seleccionan con MMR y luego se ordenan por su índice original. Esto
evita que el resumen sea una lista de oraciones en orden de puntaje, que sería
menos coherente para el lector.

### 3.6. Manejo de documentos extensos

Si hay más de 2000 oraciones, se aplica un muestreo uniforme:

```python
step = len(sentences) / limit
muestra = [sentences[int(i * step)] for i in range(limit)]
```

Se conservan oraciones de todo el documento, no solo del inicio. La interfaz
advierte que se analizaron menos oraciones que el total.

---

## 4. Relación entre la teoría y el programa

| Teoría | Código | Explicación |
| --- | --- | --- |
| Tokenización | `_TOKEN_RE` | Expresión regular Unicode que extrae letras. |
| Stopwords | `load_stopwords` | `frozenset` para búsqueda O(1). |
| Stemming | `SnowballStemmer` | Agrupa variantes morfológicas. |
| TF-IDF | `build_tfidf` | TF aumentado e IDF suavizado. |
| Similitud coseno | `build_similarity_matrix` | Producto punto de vectores normalizados. |
| Centroide | `_cosine_with_centroid` | Promedio de vectores del documento. |
| TextRank | `pagerank` | PageRank sobre grafo de oraciones. |
| MMR | `mmr_select` | Penalización por similitud con seleccionadas. |
| Jaccard | `compare_summaries` | Intersección / unión de selecciones. |
| Legibilidad | `compute_stats` | Fórmula Fernández-Huerta. |
| Detección de idioma | `detect_language` | Intersección con stopwords. |

---

## 5. Principales desafíos encontrados

### 5.1. Calidad de extracción del PDF

Los PDFs no almacenan texto de forma uniforme: pueden tener columnas, tablas,
saltos de línea arbitrarios, ligaduras o fuentes incrustadas. `pypdf` funciona
bien en documentos generados digitalmente, pero no en escaneos. Se añadieron
limpiezas específicas:

- unión de palabras cortadas con guion;
- eliminación de números de página;
- eliminación de encabezados repetidos;
- detección de PDFs sin texto para sugerir OCR.

### 5.2. Segmentación de oraciones en español

Las abreviaturas y los decimales rompen un segmentador ingenuo. Por ejemplo,
`Dr. Pérez` no debe dividirse, pero `a. m. El` sí. Se resolvió protegiendo
puntos conflictivos con un carácter privado de Unicode y aplicando reglas
especiales para abreviaturas horarias.

### 5.3. Redundancia en el resumen

Seleccionar solo las oraciones con mayor puntaje produce resúmenes repetitivos.
MMR resuelve parcialmente el problema al penalizar la similitud con las
oraciones ya elegidas.

### 5.4. Rendimiento

La matriz de similitud densa es O(N²). Para documentos largos se usa una matriz
dispersa con índice invertido y un límite de 2000 oraciones con muestreo
uniforme. Además, Streamlit cachea los resultados para evitar reprocesar el
mismo PDF.

### 5.5. Diferencia entre proporción de oraciones y de palabras

El usuario elige un porcentaje de oraciones, pero la compresión se mide en
palabras. Una oración larga puede hacer que el porcentaje de palabras sea mayor
que el porcentaje de oraciones. La interfaz muestra ambas métricas para que la
relación sea transparente.

### 5.6. Legibilidad aproximada

La fórmula Fernández-Huerta requiere contar sílabas. Se implementó una
estimación por grupos vocálicos con reglas para `qu` y `gu`. Es aproximada, pero
útil para comparar documento y resumen.

### 5.7. Manejo de errores

Cada tipo de PDF problemático tiene una excepción y un mensaje distinto. Esto
mejora la experiencia de usuario y facilita las pruebas.

---

## 6. Pruebas

Las pruebas usan `pytest` y cubren:

| Archivo | Qué valida |
| --- | --- |
| `test_preprocessing.py` | Abreviaturas, comillas, decimales, stopwords, duplicados, idioma. |
| `test_summarizers.py` | Normalización TF-IDF, cantidad de oraciones, MMR, PageRank, fallback, errores. |
| `test_keywords_metrics.py` | Palabras clave, estadísticas, compresión, sílabas. |

Ejecución:

```bash
pytest
```

Resultado actual: **19 pruebas superadas**.

---

## 7. Complejidad y rendimiento

| Etapa | Complejidad | Notas |
| --- | --- | --- |
| Extracción PDF | O(páginas × texto) | Depende de `pypdf`. |
| Segmentación | O(caracteres) | Expresiones regulares. |
| Tokenización | O(palabras) | Regex Unicode. |
| TF-IDF | O(tokens) | Conteo con `Counter`. |
| Similitud | O(Σ df²) | Matriz dispersa. |
| PageRank | O(iter × aristas) | 100 iteraciones máximo. |
| MMR | O(k × N) | k oraciones seleccionadas. |
| Métricas | O(palabras) | Conteo y sílabas. |

En un documento de 3 páginas y ~1000 palabras, el análisis completo tarda
alrededor de un segundo en un equipo portátil.

---

## 8. Seguridad y privacidad

- El PDF se procesa en memoria; no se escribe en disco salvo que el usuario
  descargue un resultado.
- No hay llamadas a APIs externas ni telemetría de Streamlit
  (`gatherUsageStats = false`).
- El límite de carga es de 25 MB y el de páginas es 400 para evitar abusos.
- Las descargas usan nombres de archivo saneados.
- No se ejecuta código contenido en el PDF.

---

## 9. Posibles extensiones

1. OCR integrado con Tesseract para PDFs escaneados.
2. Embeddings (Sentence-BERT) para similitud semántica.
3. Resumen abstractivo opcional con un LLM.
4. Evaluación ROUGE con resúmenes de referencia.
5. Exportación a DOCX y PDF.
6. Procesamiento por lotes desde la CLI.
7. Soporte para más idiomas y stopwords personalizadas.
8. Despliegue en Streamlit Community Cloud o Docker.
