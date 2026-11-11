# 📄 Resumidor de PDF con Procesamiento de Lenguaje Natural

Aplicación web que recibe un documento PDF, extrae su texto y genera un resumen
extractivo utilizando técnicas clásicas de Procesamiento de Lenguaje Natural
(PLN): **TF-IDF**, **TextRank** y **Maximal Marginal Relevance (MMR)**.

El proyecto incluye interfaz web (Streamlit), interfaz de línea de comandos,
pruebas automatizadas, un PDF de ejemplo y documentación teórica y técnica.

> **Privacidad:** todo el procesamiento ocurre en el equipo local. El PDF no se
> envía a APIs, servicios en la nube ni modelos externos.

---

## 1. Descripción general

Leer documentos extensos consume tiempo y esfuerzo. Este sistema automatiza la
tarea de identificar las oraciones más representativas de un PDF y las presenta
en un resumen breve, conservando el orden original del documento.

El flujo general es:

```text
PDF → extracción de texto → limpieza → segmentación en oraciones
    → tokenización → stopwords → stemming → vectores TF-IDF
    → ranking (TF-IDF o TextRank) → selección con MMR
    → resumen + métricas + palabras clave + descargas
```

### Características principales

- Interfaz web amigable con carga de PDF, configuración y descargas.
- Dos algoritmos de resumen: **TF-IDF + MMR** y **TextRank**.
- Control del tamaño del resumen (10 % a 50 % de las oraciones).
- Reducción de redundancia con MMR.
- Detección automática de idioma (español / inglés).
- Extracción de palabras clave mediante TF-IDF global.
- Métricas de compresión, diversidad léxica, tiempo de lectura y legibilidad
  Fernández-Huerta.
- Comparación entre ambos algoritmos mediante el índice de Jaccard.
- Exportación en **TXT**, **Markdown** y **JSON**.
- Interfaz de consola (`cli.py`) para automatizar el procesamiento.
- Manejo específico de errores: PDF cifrado, escaneado, vacío, dañado o demasiado
  extenso.
- Pruebas unitarias con `pytest`.

---

## 2. Requisitos de instalación

| Requisito | Versión recomendada |
| --- | --- |
| Python | 3.10 o superior (probado con 3.14) |
| pip | 23 o superior |
| Sistema operativo | macOS, Linux o Windows |
| Memoria RAM | 2 GB o más |
| Espacio en disco | ~200 MB para el entorno virtual |

No se necesita conexión a internet después de instalar las dependencias. No se
requieren claves de API ni descargas de corpus de NLTK.

---

## 3. Dependencias

Las dependencias de ejecución están en `requirements.txt`:

| Biblioteca | Uso en el proyecto |
| --- | --- |
| `streamlit` | Interfaz web interactiva. |
| `pypdf` | Extracción de texto y metadatos de archivos PDF. |
| `nltk` | Stemmer Snowball para español e inglés. Se usa sin descargar corpus. |
| `pandas` | Tablas, métricas y gráficos de la interfaz. |

Las dependencias de desarrollo están en `requirements-dev.txt`:

| Biblioteca | Uso |
| --- | --- |
| `pytest` | Pruebas unitarias del núcleo de PLN. |

Instalación:

```bash
pip install -r requirements.txt
# Para ejecutar las pruebas:
pip install -r requirements-dev.txt
```

---

## 4. Instrucciones de ejecución

### 4.1. Crear el entorno virtual

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2. Ejecutar la aplicación web

```bash
streamlit run app.py
```

Si el comando `streamlit` no está disponible en el PATH:

```bash
python -m streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

### 4.3. Usar la interfaz

1. Sube un PDF con texto seleccionable (no escaneado).
2. Elige el algoritmo en la barra lateral.
3. Ajusta el tamaño del resumen y, si lo deseas, las opciones avanzadas.
4. Revisa las pestañas:
   - **Resumen:** resultado y descargas.
   - **Análisis:** palabras clave, puntajes, estadísticas y comparación.
   - **Texto extraído:** contenido completo y vista por páginas.
   - **Acerca de:** explicación de los algoritmos.
5. Descarga el resultado en TXT, Markdown o JSON.

### 4.4. Ejecutar desde la consola

```bash
# Resumen con TF-IDF (20 % de las oraciones)
python cli.py resources/ejemplo_articulo.pdf

# TextRank, 15 % de las oraciones, palabras clave y estadísticas
python cli.py resources/ejemplo_articulo.pdf -m textrank -r 0.15 --keywords 10 --stats

# Guardar un informe JSON completo
python cli.py resources/ejemplo_articulo.pdf --json -o analisis.json

# Ver todas las opciones
python cli.py --help
```

Opciones principales:

| Opción | Descripción |
| --- | --- |
| `-m, --metodo` | `tfidf` o `textrank`. |
| `-r, --ratio` | Proporción de oraciones a conservar. |
| `--sin-mmr` | Desactiva la selección con diversidad. |
| `--mmr-lambda` | Balance relevancia/diversidad (0 a 1). |
| `--min-words` | Mínimo de palabras por oración. |
| `--idioma` | `auto`, `es` o `en`. |
| `--keywords N` | Muestra las N palabras clave principales. |
| `--stats` | Muestra estadísticas del documento y del resumen. |
| `--vinetas` | Presenta el resumen como lista. |
| `--json` | Emite un informe JSON. |
| `-o, --salida` | Archivo de salida. |

### 4.5. Ejecutar las pruebas

```bash
pytest
```

Resultado esperado: **19 pruebas superadas**.

### 4.6. Probar con el PDF de ejemplo

El repositorio incluye `resources/ejemplo_articulo.pdf`, un artículo de ejemplo
sobre PLN de tres páginas. También se incluye el texto fuente en
`resources/ejemplo_articulo.txt`.

---

## 5. Estructura del proyecto

```text
Proyecto_Final/
├── app.py                       # Aplicación web Streamlit
├── cli.py                       # Interfaz de línea de comandos
├── requirements.txt             # Dependencias de ejecución
├── requirements-dev.txt         # Dependencias de desarrollo
├── pytest.ini                   # Configuración de pruebas
├── README.md
├── .streamlit/
│   └── config.toml              # Tema y límite de carga
├── src/
│   ├── __init__.py
│   ├── errors.py                # Excepciones del dominio
│   ├── pdf_extractor.py         # Lectura y limpieza del PDF
│   ├── preprocessing.py         # Oraciones, tokens, stopwords y stemming
│   ├── summarizers.py           # TF-IDF, TextRank, PageRank y MMR
│   ├── keywords.py              # Extracción de palabras clave
│   ├── metrics.py               # Estadísticas y legibilidad
│   └── utils.py                 # Formatos y reportes descargables
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
    ├── 01_marco_teorico.md
    ├── 02_diseno.md
    └── 03_explicacion_tecnica.md
```

---

## 6. Cómo funciona

### 6.1. Extracción y limpieza

`src/pdf_extractor.py` usa `pypdf` para leer cada página. Luego:

- normaliza Unicode;
- une palabras cortadas con guion al final de línea;
- elimina números de página, URLs y líneas compuestas solo por símbolos;
- detecta encabezados o pies repetidos en al menos 50 % de las páginas;
- reconstruye párrafos y conserva viñetas.

Si el PDF está cifrado, dañado, vacío o no tiene texto seleccionable, se lanza
una excepción específica que la interfaz traduce en un mensaje comprensible.

### 6.2. Preprocesamiento

`src/preprocessing.py` implementa:

1. **Segmentación en oraciones** con reglas que protegen abreviaturas (`Dr.`,
   `p. ej.`), decimales (`3.14`) y comillas de cierre.
2. **Tokenización** con expresiones regulares Unicode.
3. **Normalización** a minúsculas y sin tildes (conservando la ñ).
4. **Eliminación de stopwords** desde `resources/`.
5. **Stemming** con `SnowballStemmer` de NLTK.
6. **Detección de idioma** por coincidencia de stopwords y caracteres propios.
7. **Filtrado de encabezados** y oraciones demasiado cortas.

### 6.3. Representación y resumen

`src/summarizers.py` implementa desde cero:

- **TF-IDF** con TF aumentado e IDF suavizado.
- **Similitud coseno** con matriz dispersa mediante índice invertido.
- **TextRank** con PageRank ponderado y damping de 0.85.
- **MMR** para seleccionar oraciones relevantes pero no redundantes.

El resumen final se reordena según la posición original de las oraciones para
mantener la coherencia del documento.

### 6.4. Palabras clave y métricas

- `src/keywords.py`: ranking TF-IDF global de términos, agrupando variantes por
  stem y mostrando la forma superficial más frecuente.
- `src/metrics.py`: palabras, oraciones, caracteres, diversidad léxica, sílabas,
  tiempo de lectura e índice de legibilidad Fernández-Huerta.

---

## 7. Algoritmos implementados

### TF-IDF

```text
tf-idf(t, d) = (0.5 + 0.5 · tf(t,d) / max_tf(d)) · (log((N+1)/(df(t)+1)) + 1)
```

Cada oración se normaliza L2 y se compara con el centroide del documento
mediante similitud coseno. Las oraciones más cercanas al centroide representan
mejor el contenido global.

### TextRank

Cada oración es un nodo de un grafo ponderado; el peso de la arista es la
similitud coseno entre los vectores TF-IDF. PageRank calcula la centralidad:

```text
PR(i) = (1-d)/N + d · Σ_j (w_ji / Σ_k w_jk) · PR(j)
```

### MMR

```text
MMR(i) = λ · relevancia(i) - (1-λ) · max similitud(i, seleccionadas)
```

Con λ = 0.7 se favorece la relevancia sin repetir ideas ya cubiertas.

### Métricas de evaluación

- **Compresión:** `1 - palabras_resumen / palabras_original`.
- **Jaccard:** coincidencia entre las oraciones elegidas por dos métodos.
- **Fernández-Huerta:** legibilidad estimada a partir de sílabas y longitud de
  oración.

---

## 8. Manejo de errores

| Situación | Mensaje / acción |
| --- | --- |
| Archivo que no es PDF | `UnsupportedFileError` con validación de cabecera `%PDF`. |
| PDF protegido | `PDFEncryptedError`; se solicita retirar la contraseña. |
| PDF escaneado | `PDFScannedError`; se sugiere aplicar OCR. |
| PDF vacío o con muy poco texto | `PDFEmptyError`. |
| Más de 400 páginas | `PDFTooLargeError`. |
| Texto sin oraciones aprovechables | Aviso en la interfaz y texto extraído disponible. |
| Error inesperado | Mensaje amigable y detalle técnico en un expander. |
| Archivo mayor a 25 MB | Validación en la interfaz antes de procesar. |

---

## 9. Capturas de pantalla (opcional)

La interfaz incluye cuatro pestañas: **Resumen**, **Análisis**, **Texto extraído**
y **Acerca de**. Para documentar el proyecto se recomienda capturar:

1. Pantalla inicial con el cargador de PDF.
2. Resumen generado y métricas principales.
3. Pestaña de análisis con palabras clave y puntajes por oración.
4. Comparación TF-IDF vs. TextRank.

Guarda las imágenes en `docs/capturas/` y enlázalas aquí, por ejemplo:

```markdown
![Pantalla inicial](docs/capturas/inicio.png)
![Resumen generado](docs/capturas/resumen.png)
```

---

## 10. Limitaciones y trabajo futuro

- El resumen es **extractivo**: no redacta oraciones nuevas.
- Los PDF escaneados requieren OCR previo.
- La extracción de PDFs con columnas complejas o tablas puede ser imperfecta.
- La detección de idioma es una heurística basada en stopwords.
- La estimación de sílabas es aproximada.
- El algoritmo no utiliza modelos semánticos profundos (embeddings o LLM).

Posibles mejoras:

- Añadir OCR integrado con Tesseract.
- Incorporar embeddings y similitud semántica.
- Exportar a DOCX o PDF.
- Añadir resumen abstractivo opcional con un LLM.
- Evaluar con ROUGE frente a resúmenes de referencia.
- Soportar más idiomas.

---

## 11. Documentación adicional

- [Marco teórico](docs/01_marco_teorico.md)
- [Diseño de la solución](docs/02_diseno.md)
- [Explicación técnica](docs/03_explicacion_tecnica.md)

---

## 12. Contexto académico

Proyecto final de **Procesamiento de Lenguaje Natural** para la asignatura
**Autómatas 2**. El objetivo es aplicar conceptos de procesamiento de texto,
tokenización, representación vectorial, análisis y extracción de información en
una aplicación funcional.
