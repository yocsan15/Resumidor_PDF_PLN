# Marco teórico

## 1. Definición del problema

### 1.1. Contexto

La cantidad de información disponible en formato digital crece más rápido que la
capacidad humana de leerla completa. Artículos científicos, informes técnicos,
documentos administrativos y noticias suelen tener decenas o cientos de páginas.
Un lector necesita identificar rápidamente las ideas principales, pero hacerlo
manualmente consume tiempo y es propenso a omitir información relevante.

### 1.2. Problema

¿Cómo construir una aplicación que reciba un documento PDF y genere
automáticamente un resumen breve, coherente y representativo de su contenido,
utilizando técnicas de Procesamiento de Lenguaje Natural sin depender de
servicios externos ni de modelos de lenguaje propietarios?

### 1.3. Objetivo general

Desarrollar un resumidor de PDF que aplique técnicas clásicas de PLN para
extraer las oraciones más relevantes de un documento y presentarlas en una
interfaz web amigable.

### 1.4. Objetivos específicos

1. Extraer y limpiar el texto de archivos PDF.
2. Segmentar el texto en oraciones y tokens.
3. Representar las oraciones mediante vectores TF-IDF.
4. Implementar dos algoritmos de ranking: TF-IDF y TextRank.
5. Reducir la redundancia del resumen con MMR.
6. Extraer palabras clave y calcular métricas del documento y del resumen.
7. Diseñar una interfaz web y una interfaz de consola.
8. Manejar errores comunes de archivos PDF.
9. Validar el funcionamiento con pruebas automatizadas.

### 1.5. Alcance

El sistema genera resúmenes **extractivos**: selecciona oraciones del documento
original. No redacta texto nuevo ni interpreta el significado profundo del
documento. El alcance incluye documentos PDF con texto seleccionable en español
o inglés; los documentos escaneados requieren OCR previo.

---

## 2. Fundamentos teóricos

### 2.1. Procesamiento de Lenguaje Natural

El Procesamiento de Lenguaje Natural (PLN) es una disciplina que combina
lingüística, informática y estadística para permitir que las computadoras
analicen, comprendan y generen lenguaje humano. Se organiza en varios niveles:

| Nivel | Unidad de análisis | Ejemplos de tareas |
| --- | --- | --- |
| Morfológico | Palabra | Tokenización, stemming, lematización |
| Sintáctico | Oración | Análisis gramatical, segmentación |
| Semántico | Significado | Similitud, desambiguación |
| Pragmático | Contexto | Análisis de discurso, resumen |

Este proyecto trabaja principalmente en los niveles morfológico y sintáctico,
con una representación vectorial que aproxima la semántica mediante la
distribución de palabras.

### 2.2. Resumen extractivo y abstractivo

- **Extractivo:** selecciona fragmentos literales del documento. Es más fácil de
  implementar, conserva la fidelidad del texto y no requiere un modelo generativo.
- **Abstractivo:** redacta un texto nuevo que sintetiza el contenido. Requiere
  modelos de lenguaje y puede introducir información no presente en el original.

El proyecto implementa resumen extractivo porque permite explicar cada paso del
algoritmo, no necesita entrenamiento y funciona completamente en local.

### 2.3. Preprocesamiento de texto

Antes de aplicar cualquier algoritmo, el texto debe normalizarse:

1. **Limpieza:** eliminar caracteres de control, encabezados, pies de página,
   URLs y numeración.
2. **Segmentación:** dividir el texto en oraciones.
3. **Tokenización:** dividir cada oración en palabras o tokens.
4. **Normalización:** convertir a minúsculas y, opcionalmente, eliminar tildes.
5. **Stopwords:** eliminar palabras muy frecuentes con poco contenido semántico.
6. **Stemming:** reducir las palabras a una raíz común.

La segmentación de oraciones es especialmente difícil en español por las
abreviaturas (`Dr.`, `Sra.`, `p. ej.`), los decimales (`3.14`) y los signos de
apertura (`¿`, `¡`). El proyecto usa reglas que protegen estos casos antes de
buscar los signos de cierre.

### 2.4. Modelo de espacio vectorial

El modelo de espacio vectorial representa cada documento u oración como un
vector en un espacio de dimensiones igual al vocabulario. Una forma sencilla es
la **bolsa de palabras**, que cuenta la frecuencia de cada término e ignora el
orden.

La bolsa de palabras tiene dos limitaciones:

- asigna la misma importancia a todas las palabras;
- favorece las oraciones largas, que contienen más términos.

TF-IDF corrige parcialmente estos problemas.

### 2.5. TF-IDF

TF-IDF combina dos medidas:

- **Frecuencia de término (TF):** cuántas veces aparece un término en una
  oración.
- **Frecuencia inversa de documento (IDF):** qué tan raro es el término en el
  conjunto de oraciones.

```text
TF-IDF(t, d) = TF(t, d) · IDF(t)
IDF(t) = log((N + 1) / (df(t) + 1)) + 1
```

donde `N` es el número de oraciones y `df(t)` es el número de oraciones que
contienen el término `t`.

En la implementación se usa TF aumentado para no penalizar en exceso las
oraciones largas:

```text
TF_aumentado(t, d) = 0.5 + 0.5 · tf(t, d) / max_tf(d)
```

El IDF suavizado evita divisiones por cero y reduce el peso de términos que
aparecen en todas las oraciones.

### 2.6. Similitud coseno

La similitud coseno mide el ángulo entre dos vectores:

```text
sim(A, B) = (A · B) / (||A|| · ||B||)
```

Su valor está entre 0 y 1 cuando los componentes son no negativos. Es
independiente de la longitud de los vectores, por lo que no favorece
directamente a las oraciones largas. Se usa para comparar oraciones entre sí y
con el centroide del documento.

### 2.7. Centroide del documento

El centroide es el vector promedio de todas las oraciones. Una oración cercana
al centroide comparte vocabulario con muchas otras y, por tanto, es
representativa del contenido global. En el método TF-IDF, la relevancia de una
oración se calcula como su similitud coseno con el centroide.

### 2.8. TextRank y PageRank

TextRank adapta PageRank al texto. Cada oración es un nodo de un grafo y las
aristas representan la similitud entre oraciones. La importancia de un nodo
depende de la importancia de sus vecinos:

```text
PR(i) = (1 - d) / N + d · Σ_j (w_ji / Σ_k w_jk) · PR(j)
```

donde `d` es el factor de amortiguamiento (0.85), `N` es el número de nodos y
`w_ji` es el peso de la arista entre `j` e `i`.

El algoritmo itera hasta que los puntajes convergen o se alcanza un máximo de
iteraciones. Las oraciones con muchas conexiones con otras oraciones importantes
reciben un puntaje alto.

### 2.9. Maximal Marginal Relevance (MMR)

Un resumen puede contener oraciones relevantes pero repetitivas. MMR selecciona
en cada paso la oración que maximiza una combinación de relevancia y novedad:

```text
MMR(i) = λ · relevancia(i) - (1 - λ) · max_{j ∈ S} similitud(i, j)
```

donde `S` es el conjunto de oraciones ya seleccionadas. Con `λ = 0.7` se prioriza
la relevancia, pero se penalizan las oraciones muy parecidas a las elegidas.

### 2.10. Evaluación de resúmenes

La evaluación automática de resúmenes es un problema abierto. Algunas métricas
habituales son:

- **ROUGE:** mide la cobertura de palabras o secuencias del resumen de
  referencia.
- **BLEU:** compara secuencias de palabras; se usa más en traducción.
- **Compresión:** porcentaje de reducción de palabras.
- **Legibilidad:** estima la dificultad de lectura.

En este proyecto se reportan compresión, tiempo de lectura, diversidad léxica y
legibilidad. Además, se compara la coincidencia entre algoritmos con el índice de
Jaccard:

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

### 2.11. Legibilidad Fernández-Huerta

El índice de Fernández-Huerta adapta el índice de Flesch al español:

```text
FH = 206.84 - 0.60 · (sílabas por 100 palabras) - 1.02 · (palabras por oración)
```

El resultado se interpreta cualitativamente: muy fácil, fácil, normal, difícil,
etc. La estimación de sílabas del proyecto cuenta grupos vocálicos y aplica
reglas básicas para `qu` y `gu`, por lo que es aproximada.

### 2.12. Detección de idioma

El sistema compara la cantidad de stopwords distintas de español e inglés que
aparecen en una muestra del texto, y añade señales tipográficas como `ñ`, `¿` e
`¡`. Es una heurística suficiente para elegir stopwords y stemmer, no un
detector lingüístico profesional.

---

## 3. Conceptos involucrados

| Concepto | Definición |
| --- | --- |
| **Token** | Unidad mínima de texto; normalmente una palabra. |
| **Stopword** | Palabra muy frecuente con poco contenido semántico. |
| **Stemming** | Reducción de una palabra a su raíz. |
| **Lematización** | Reducción a la forma canónica usando morfología. |
| **Corpus** | Conjunto de documentos usado para análisis. |
| **Vocabulario** | Conjunto de términos distintos. |
| **DF** | Número de documentos/oraciones que contienen un término. |
| **IDF** | Peso que penaliza los términos frecuentes. |
| **Centroide** | Vector promedio de un conjunto de vectores. |
| **Similitud coseno** | Medida del ángulo entre dos vectores. |
| **Grafo** | Estructura de nodos y aristas; aquí, oraciones y similitudes. |
| **PageRank** | Algoritmo de centralidad en grafos. |
| **MMR** | Selección que combina relevancia y diversidad. |
| **Jaccard** | Medida de coincidencia entre dos conjuntos. |
| **Resumen extractivo** | Resumen formado por fragmentos del original. |
| **Resumen abstractivo** | Resumen redactado con palabras nuevas. |
| **Legibilidad** | Facilidad con la que un texto puede leerse. |

---

## 4. Relación entre teoría y proyecto

| Concepto teórico | Implementación |
| --- | --- |
| Tokenización | `_TOKEN_RE` en `src/preprocessing.py`. |
| Stopwords | `resources/stopwords_es.txt` y `stopwords_en.txt`. |
| Stemming | `SnowballStemmer` en `src/preprocessing.py`. |
| TF-IDF | `build_tfidf` en `src/summarizers.py`. |
| Similitud coseno | `build_similarity_matrix` y `_cosine_with_centroid`. |
| TextRank / PageRank | `pagerank` en `src/summarizers.py`. |
| MMR | `mmr_select` en `src/summarizers.py`. |
| Evaluación | `src/metrics.py` y `compare_summaries`. |
| Legibilidad | `compute_stats` en `src/metrics.py`. |

---

## 5. Referencias

- Jurafsky, D. y Martin, J. H. (2023). *Speech and Language Processing*.
  Pearson.
- Manning, C. D. y Schütze, H. (1999). *Foundations of Statistical Natural
  Language Processing*. MIT Press.
- Mihalcea, R. y Tarau, P. (2004). *TextRank: Bringing Order into Texts*.
  Proceedings of EMNLP.
- Carbonell, J. y Goldstein, J. (1998). *The Use of MMR, Diversity-Based
  Reranking for Reordering Documents*. SIGIR.
- Fernández-Huerta, J. (1959). *Medidas sencillas de lecturabilidad*.
  Consigna.
