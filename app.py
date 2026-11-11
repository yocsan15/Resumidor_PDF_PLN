"""Aplicación web del Resumidor de PDF con PLN.

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.errors import PDFSummarizerError
from src.keywords import Keyword, extract_keywords
from src.metrics import (
    TextStats,
    compression_ratio,
    compute_stats,
    reading_time_saved,
)
from src.pdf_extractor import PDFDocument, extract_text_from_pdf
from src.preprocessing import Sentence, detect_language, preprocess_text
from src.summarizers import (
    METHOD_LABELS,
    SummaryResult,
    compare_summaries,
    summarize,
)
from src.utils import (
    build_json_report,
    build_markdown_report,
    format_summary,
    human_readable_size,
    reading_time_label,
    report_to_json,
    safe_filename,
    selected_sentence_rows,
)

MAX_FILE_MB = 25

st.set_page_config(
    page_title="Resumidor de PDF con PLN",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 1250px; }
        .hero {
            background: linear-gradient(135deg, #5B4BDB 0%, #8B7CF6 55%, #B8A9FF 100%);
            color: #FFFFFF;
            padding: 2rem 2.2rem;
            border-radius: 20px;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 30px rgba(91, 75, 219, 0.20);
        }
        .hero h1 { margin: 0 0 0.45rem 0; font-size: 2.15rem; line-height: 1.2; }
        .hero p { margin: 0; opacity: 0.94; font-size: 1.04rem; max-width: 820px; }
        .step-card {
            border: 1px solid #E7E3FF;
            background: #FAF9FF;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }
        .step-card h4 { margin-top: 0; color: #4C3FC4; }
        .small-muted { color: #6B7280; font-size: 0.9rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Funciones con caché: Streamlit evita repetir el trabajo si no cambian las
# entradas (mismo PDF, mismo texto o mismos parámetros).
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def cached_extract(data: bytes, source_name: str) -> PDFDocument:
    return extract_text_from_pdf(data, source_name=source_name)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_detect_language(text: str) -> str:
    return detect_language(text)


@st.cache_data(show_spinner=False, max_entries=32)
def analyze_document(
    text: str,
    language: str,
    method: str,
    ratio: float,
    use_mmr: bool,
    mmr_lambda: float,
    min_words: int,
    top_n: int,
) -> tuple[
    list[Sentence],
    SummaryResult | None,
    list[Keyword],
    TextStats,
    TextStats | None,
]:
    sentences = preprocess_text(text, language=language, min_words=min_words)
    original_stats = compute_stats(text)

    if not sentences:
        return sentences, None, [], original_stats, None

    result = summarize(
        sentences,
        method=method,
        ratio=ratio,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
    )
    keywords = extract_keywords(sentences, top_n=top_n)
    summary_stats = compute_stats(result.summary)
    return sentences, result, keywords, original_stats, summary_stats


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.caption("Ajusta el resumen antes de subir el documento.")

    method_label = st.radio(
        "Algoritmo de resumen",
        options=list(METHOD_LABELS.values()),
        index=0,
        help=(
            "TF-IDF + MMR prioriza oraciones representativas y evita repetir ideas. "
            "TextRank usa un grafo de similitud y PageRank."
        ),
    )
    method = next(
        key for key, value in METHOD_LABELS.items() if value == method_label
    )

    ratio_pct = st.slider(
        "Tamaño del resumen",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        format="%d%%",
        help="Porcentaje de oraciones del documento que se conservarán.",
    )
    ratio = ratio_pct / 100.0

    language_option = st.selectbox(
        "Idioma del documento",
        options=["Detección automática", "Español", "English"],
        index=0,
        help="El idioma define las stopwords y el stemmer utilizados.",
    )

    with st.expander("Opciones avanzadas"):
        use_mmr = st.checkbox(
            "Evitar redundancia (MMR)",
            value=True,
            help="Penaliza oraciones parecidas a las ya seleccionadas.",
        )
        mmr_lambda = st.slider(
            "Relevancia ↔ diversidad",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            disabled=not use_mmr,
            help="Valores altos priorizan relevancia; valores bajos, diversidad.",
        )
        min_words = st.slider(
            "Mínimo de palabras por oración",
            min_value=4,
            max_value=15,
            value=6,
            help="Descarta encabezados, pies de página y fragmentos muy cortos.",
        )
        top_n = st.slider(
            "Palabras clave a mostrar",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
        )
        bullet_format = st.checkbox("Mostrar resumen en viñetas", value=False)

    st.divider()
    st.caption(
        "🔒 El procesamiento es local: el PDF no se envía a servicios externos."
    )


# ---------------------------------------------------------------------------
# Encabezado y carga del archivo
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📄 Resumidor de PDF con PLN</h1>
        <p>
            Extrae las ideas principales de documentos extensos con técnicas de
            Procesamiento de Lenguaje Natural: TF-IDF, TextRank y MMR.
            Sin APIs, sin costo por token y sin enviar tus archivos a la nube.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Selecciona un PDF con texto seleccionable",
    type=["pdf"],
    help=f"Tamaño máximo: {MAX_FILE_MB} MB. Los PDF escaneados necesitan OCR.",
)


if uploaded is None:
    st.markdown("### ¿Cómo funciona?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="step-card">
                <h4>1. Sube tu PDF</h4>
                <p>El sistema extrae el texto página por página y limpia
                encabezados, numeración y guiones de corte.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="step-card">
                <h4>2. Procesa el lenguaje</h4>
                <p>Tokeniza, elimina stopwords, aplica stemming y calcula la
                importancia de cada oración y palabra.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="step-card">
                <h4>3. Obtén el resumen</h4>
                <p>Descarga el resultado en TXT, Markdown o JSON junto con
                métricas, palabras clave y el análisis del documento.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "💡 **Consejo:** si el PDF es un escaneo (una imagen por página), primero "
        "aplica OCR con una herramienta como OCRmyPDF. Sin texto seleccionable "
        "no es posible aplicar PLN clásico."
    )
    st.markdown("#### Algoritmos disponibles")
    st.markdown(
        "- **TF-IDF + MMR:** mide qué tan representativa es cada oración y "
        "selecciona un conjunto diverso.\n"
        "- **TextRank:** construye un grafo de similitud entre oraciones y "
        "aplica PageRank para encontrar las más centrales.\n"
        "- **Palabras clave:** ranking TF-IDF global de términos del documento."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Validación y extracción
# ---------------------------------------------------------------------------
data = uploaded.getvalue()

if len(data) > MAX_FILE_MB * 1024 * 1024:
    st.error(
        f"El archivo pesa {human_readable_size(len(data))} y el límite es "
        f"{MAX_FILE_MB} MB. Comprime el PDF o divídelo en partes."
    )
    st.stop()

try:
    with st.spinner("Extrayendo texto del PDF..."):
        document = cached_extract(data, uploaded.name)
except PDFSummarizerError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # pragma: no cover - red de seguridad de la interfaz
    st.error("Ocurrió un error inesperado al leer el PDF.")
    with st.expander("Detalle técnico"):
        st.exception(exc)
    st.stop()

for warning in document.warnings:
    st.warning(warning)

if language_option == "Detección automática":
    language = cached_detect_language(document.text)
else:
    language = "spanish" if language_option == "Español" else "english"
language_name = "Español" if language == "spanish" else "English"

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(f"### {document.title}")
    st.caption(
        f"{uploaded.name} · {human_readable_size(len(data))} · "
        f"{document.pages} páginas · Idioma: {language_name}"
    )
with header_right:
    st.metric("Páginas", document.pages)

try:
    with st.spinner("Aplicando técnicas de PLN..."):
        sentences, result, keywords, original_stats, summary_stats = analyze_document(
            document.text,
            language,
            method,
            ratio,
            use_mmr,
            mmr_lambda,
            min_words,
            top_n,
        )
except Exception as exc:  # pragma: no cover
    st.error("No se pudo procesar el texto del documento.")
    with st.expander("Detalle técnico"):
        st.exception(exc)
    st.stop()

if result is None or summary_stats is None:
    st.warning(
        "No se encontraron oraciones suficientemente largas para resumir. "
        "Prueba a bajar el mínimo de palabras por oración o revisa el texto extraído."
    )
    st.text_area("Texto extraído", document.text, height=320)
    st.stop()

if len(sentences) < 3:
    st.info(
        "El documento contiene muy pocas oraciones aprovechables; el resumen "
        "puede ser casi igual al texto original."
    )

if result.config.get("truncated"):
    st.warning(
        f"El documento es muy extenso: se analizaron {result.config['sentence_count']} "
        f"de {result.config['original_sentence_count']} oraciones mediante un "
        "muestreo uniforme para mantener un tiempo de respuesta razonable."
    )

if result.config.get("fallback_to_tfidf"):
    st.info(
        "Las oraciones no comparten vocabulario suficiente para construir el grafo "
        "de TextRank; se usó TF-IDF como respaldo."
    )

compression = compression_ratio(original_stats, summary_stats)
saved_time = reading_time_saved(original_stats, summary_stats)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Palabras originales", f"{original_stats.words:,}")
metric2.metric("Palabras del resumen", f"{summary_stats.words:,}")
metric3.metric("Compresión", f"{compression:.0%}")
metric4.metric("Ahorro de lectura", reading_time_label(saved_time))

base_name = safe_filename(uploaded.name)
summary_text = format_summary(result, bullets=bullet_format)

tab_summary, tab_analysis, tab_text, tab_about = st.tabs(
    ["📝 Resumen", "📊 Análisis", "📄 Texto extraído", "ℹ️ Acerca de"]
)


# ---------------------------------------------------------------------------
# Pestaña: resumen
# ---------------------------------------------------------------------------
with tab_summary:
    st.markdown(
        f"**{len(result.selected)} oraciones seleccionadas** de "
        f"{len(result.sentences)} analizadas · {result.method_label} · "
        f"{ratio_pct}% objetivo"
    )

    with st.container(border=True):
        st.markdown(summary_text)

    st.caption(
        f"Tiempo de lectura estimado del resumen: "
        f"{reading_time_label(summary_stats.reading_minutes)} "
        f"(documento completo: {reading_time_label(original_stats.reading_minutes)})."
    )

    download1, download2, download3 = st.columns(3)
    download1.download_button(
        "📥 Descargar TXT",
        data=summary_text,
        file_name=f"{base_name}_resumen.txt",
        mime="text/plain",
    )

    markdown_report = build_markdown_report(
        document,
        result,
        original_stats,
        summary_stats,
        keywords,
        language,
        uploaded.name,
    )
    download2.download_button(
        "📥 Descargar Markdown",
        data=markdown_report,
        file_name=f"{base_name}_resumen.md",
        mime="text/markdown",
    )

    json_report = build_json_report(
        document,
        result,
        original_stats,
        summary_stats,
        keywords,
        language,
        uploaded.name,
    )
    download3.download_button(
        "📥 Descargar JSON",
        data=report_to_json(json_report),
        file_name=f"{base_name}_analisis.json",
        mime="application/json",
    )

    with st.expander("Ver oraciones seleccionadas y sus puntajes"):
        st.dataframe(
            pd.DataFrame(selected_sentence_rows(result)),
            hide_index=True,
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Pestaña: análisis
# ---------------------------------------------------------------------------
with tab_analysis:
    st.subheader("Palabras clave")
    if keywords:
        keyword_df = pd.DataFrame(
            [
                {
                    "Término": keyword.term,
                    "Puntaje": keyword.score,
                    "Frecuencia": keyword.frequency,
                }
                for keyword in keywords
            ]
        )
        st.bar_chart(
            keyword_df.set_index("Término")["Puntaje"],
            height=320,
        )
        st.dataframe(keyword_df, hide_index=True, width="stretch")
    else:
        st.info("No se encontraron palabras clave.")

    st.divider()
    st.subheader("Puntaje de cada oración")
    selected_set = set(result.selected)
    score_df = pd.DataFrame(
        {
            "Oración": list(range(1, len(result.sentences) + 1)),
            "Puntaje": [round(score, 3) for score in result.scores],
            "Estado": [
                "Seleccionada" if index in selected_set else "Descartada"
                for index in range(len(result.sentences))
            ],
        }
    )
    try:
        st.bar_chart(
            score_df,
            x="Oración",
            y="Puntaje",
            color="Estado",
            height=320,
        )
    except Exception:
        st.bar_chart(score_df.set_index("Oración")["Puntaje"], height=320)
    st.caption(
        "Cada barra representa la relevancia relativa de una oración. "
        "Las oraciones seleccionadas se reordenan según su posición original."
    )

    st.divider()
    st.subheader("Estadísticas del texto")
    stats_df = pd.DataFrame(
        {
            "Métrica": [
                "Palabras",
                "Oraciones (segmentadas)",
                "Caracteres",
                "Palabras únicas",
                "Diversidad léxica",
                "Palabras por oración",
                "Sílabas (estimadas)",
                "Tiempo de lectura",
            ],
            "Original": [
                f"{original_stats.words:,}",
                f"{original_stats.sentences:,}",
                f"{original_stats.characters:,}",
                f"{original_stats.unique_words:,}",
                f"{original_stats.lexical_diversity:.2f}",
                f"{original_stats.average_sentence_length:.1f}",
                f"{original_stats.syllables:,}",
                reading_time_label(original_stats.reading_minutes),
            ],
            "Resumen": [
                f"{summary_stats.words:,}",
                f"{summary_stats.sentences:,}",
                f"{summary_stats.characters:,}",
                f"{summary_stats.unique_words:,}",
                f"{summary_stats.lexical_diversity:.2f}",
                f"{summary_stats.average_sentence_length:.1f}",
                f"{summary_stats.syllables:,}",
                reading_time_label(summary_stats.reading_minutes),
            ],
        }
    )
    st.dataframe(stats_df, hide_index=True, width="stretch")
    st.caption(
        "Las oraciones segmentadas incluyen encabezados y fragmentos cortos; "
        "el resumen solo considera las oraciones aprovechables."
    )

    readability1, readability2 = st.columns(2)
    with readability1:
        st.metric(
            "Legibilidad del documento (Fernández-Huerta)",
            f"{original_stats.fernandez_huerta:.1f}",
        )
        st.caption(original_stats.readability_label)
    with readability2:
        st.metric(
            "Legibilidad del resumen (Fernández-Huerta)",
            f"{summary_stats.fernandez_huerta:.1f}",
        )
        st.caption(summary_stats.readability_label)

    st.divider()
    st.subheader("Comparación de algoritmos")
    if st.checkbox("Calcular también el método alternativo", key="compare_methods"):
        other_method = "textrank" if method == "tfidf" else "tfidf"
        with st.spinner(f"Calculando {METHOD_LABELS[other_method]}..."):
            _, other_result, _, _, _ = analyze_document(
                document.text,
                language,
                other_method,
                ratio,
                use_mmr,
                mmr_lambda,
                min_words,
                top_n,
            )

        if other_result is None:
            st.info("El método alternativo no pudo generar un resumen.")
        else:
            comparison = compare_summaries(result, other_result)
            compare1, compare2, compare3 = st.columns(3)
            compare1.metric(
                "Coincidencia (Jaccard)",
                f"{comparison['jaccard']:.0%}",
            )
            compare2.metric("Oraciones en común", comparison["common"])
            compare3.metric(
                "Oraciones distintas",
                comparison["union"] - comparison["common"],
            )

            column_a, column_b = st.columns(2)
            with column_a:
                st.markdown(f"**{result.method_label}**")
                st.write(format_summary(result))
            with column_b:
                st.markdown(f"**{other_result.method_label}**")
                st.write(format_summary(other_result))


# ---------------------------------------------------------------------------
# Pestaña: texto extraído
# ---------------------------------------------------------------------------
with tab_text:
    st.subheader("Texto extraído del PDF")
    st.caption(
        f"{len(document.text):,} caracteres · {document.pages} páginas. "
        "Este es el texto que se usó para el análisis."
    )
    st.text_area(
        "Contenido completo",
        document.text,
        height=420,
        label_visibility="collapsed",
    )
    st.download_button(
        "📥 Descargar texto completo",
        data=document.text,
        file_name=f"{base_name}_texto.txt",
        mime="text/plain",
    )

    with st.expander("Ver el texto página por página"):
        page_number = st.selectbox(
            "Página",
            options=list(range(1, document.pages + 1)),
            index=0,
        )
        st.text_area(
            f"Texto de la página {page_number}",
            document.page_texts[page_number - 1],
            height=300,
        )


# ---------------------------------------------------------------------------
# Pestaña: acerca de
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("¿Cómo funciona el resumidor?")
    st.markdown(
        """
        1. **Extracción:** se lee el PDF con `pypdf`, se unen palabras cortadas
           por guion y se eliminan números de página y encabezados repetidos.
        2. **Preprocesamiento:** se segmenta en oraciones, se tokeniza, se
           eliminan stopwords y se aplica *stemming* con Snowball.
        3. **Representación:** cada oración se convierte en un vector TF-IDF
           normalizado.
        4. **Selección:** se puntúan las oraciones con TF-IDF o TextRank y se
           eligen las mejores con MMR para evitar redundancia.
        5. **Presentación:** se reordenan las oraciones según el documento
           original y se generan métricas, palabras clave y descargas.
        """
    )

    st.markdown("#### Fórmulas principales")
    st.latex(
        r"\text{tf-idf}(t,d)=\left(0.5+0.5\frac{f_{t,d}}{\max f_{.,d}}\right)"
        r"\left(\log\frac{N+1}{df_t+1}+1\right)"
    )
    st.latex(
        r"\text{MMR}(i)=\lambda\cdot\text{relevancia}(i)"
        r"-(1-\lambda)\cdot\max_{j\in S}\text{sim}(i,j)"
    )
    st.latex(
        r"PR(i)=\frac{1-d}{N}+d\sum_{j}\frac{w_{ji}}{\sum_k w_{jk}}PR(j)"
    )

    st.markdown("#### Relación con el curso")
    st.markdown(
        "- **Procesamiento de texto:** normalización, limpieza y segmentación.\n"
        "- **Tokenización:** expresiones regulares Unicode y stopwords.\n"
        "- **Análisis y extracción:** TF-IDF, similitud coseno, TextRank/PageRank "
        "y extracción de palabras clave.\n"
        "- **Estructuras de datos:** listas, tuplas, diccionarios, conjuntos, "
        "matriz dispersa y grafo ponderado."
    )

    st.info(
        "🔒 **Privacidad:** el PDF se procesa en memoria y no se envía a "
        "ningún servicio externo. Las descargas se generan en tu equipo."
    )

    st.markdown("#### Limitaciones")
    st.markdown(
        "- El resumen es **extractivo**: conserva oraciones del original, no "
        "redacta texto nuevo.\n"
        "- Los PDF escaneados o con columnas complejas pueden requerir OCR o "
        "un extractor más avanzado.\n"
        "- La detección de idioma es una heurística basada en stopwords.\n"
        "- La legibilidad Fernández-Huerta usa una estimación de sílabas."
    )

st.divider()
st.caption(
    "Proyecto final · Procesamiento de Lenguaje Natural · "
    "TF-IDF, TextRank y MMR implementados en Python."
)
