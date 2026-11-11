"""Funciones auxiliares de formato y exportación."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .keywords import Keyword
from .metrics import TextStats, compression_ratio
from .pdf_extractor import PDFDocument
from .summarizers import SummaryResult


def human_readable_size(num_bytes: int) -> str:
    """Convierte bytes a una cadena legible (KB, MB)."""

    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024**2:.1f} MB"


def reading_time_label(minutes: float) -> str:
    """Formatea minutos de lectura de forma amigable."""

    if minutes < 1:
        seconds = max(1, round(minutes * 60))
        return f"{seconds} s"
    return f"{minutes:.1f} min"


def safe_filename(name: str, fallback: str = "resumen") -> str:
    """Genera un nombre de archivo seguro para las descargas."""

    stem = Path(name).stem if name else fallback
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return (stem or fallback)[:80]


def _ensure_period(sentence: str) -> str:
    sentence = sentence.strip()
    if sentence and sentence[-1] not in ".!?…":
        return sentence + "."
    return sentence


def format_summary(result: SummaryResult, bullets: bool = False) -> str:
    """Devuelve el resumen como párrafo o como lista de viñetas."""

    texts = [_ensure_period(sentence.text) for sentence in result.selected_sentences]
    if bullets:
        return "\n".join(f"- {text}" for text in texts)
    return " ".join(texts)


def selected_sentence_rows(result: SummaryResult) -> list[dict[str, object]]:
    """Filas para la tabla de oraciones seleccionadas."""

    rows: list[dict[str, object]] = []
    for order, index in enumerate(sorted(result.selected), start=1):
        sentence = result.sentences[index]
        rows.append(
            {
                "#": order,
                "Posición": sentence.index + 1,
                "Puntaje": round(result.scores[index], 3),
                "Oración": sentence.text,
            }
        )
    return rows


def build_markdown_report(
    document: PDFDocument,
    result: SummaryResult,
    original_stats: TextStats,
    summary_stats: TextStats,
    keywords: list[Keyword],
    language: str,
    source_name: str,
) -> str:
    """Construye un informe Markdown descargable."""

    language_name = "Español" if language == "spanish" else "English"
    compression = compression_ratio(original_stats, summary_stats)
    title = document.title or Path(source_name).stem or "Documento"

    lines: list[str] = [
        "# Resumen del documento",
        "",
        f"**Archivo:** {source_name or document.source_name or 'PDF'}",
        f"**Título:** {title}",
        f"**Páginas:** {document.pages}",
        f"**Idioma:** {language_name}",
        f"**Método:** {result.method_label}",
        f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Compresión:** {compression:.0%}",
        "",
        "## Resumen",
        "",
        format_summary(result),
        "",
        "## Palabras clave",
        "",
    ]

    if keywords:
        lines.extend(f"- {keyword.term} ({keyword.score:.2f})" for keyword in keywords)
    else:
        lines.append("_No se encontraron palabras clave._")

    lines.extend(
        [
            "",
            "## Estadísticas",
            "",
            "| Métrica | Documento original | Resumen |",
            "| --- | ---: | ---: |",
            f"| Palabras | {original_stats.words:,} | {summary_stats.words:,} |",
            f"| Oraciones | {original_stats.sentences:,} | {summary_stats.sentences:,} |",
            f"| Caracteres | {original_stats.characters:,} | {summary_stats.characters:,} |",
            f"| Diversidad léxica | {original_stats.lexical_diversity:.2f} | {summary_stats.lexical_diversity:.2f} |",
            f"| Palabras por oración | {original_stats.average_sentence_length:.1f} | {summary_stats.average_sentence_length:.1f} |",
            f"| Legibilidad (F. Huerta) | {original_stats.fernandez_huerta:.1f} ({original_stats.readability_label}) | {summary_stats.fernandez_huerta:.1f} ({summary_stats.readability_label}) |",
            f"| Tiempo de lectura | {reading_time_label(original_stats.reading_minutes)} | {reading_time_label(summary_stats.reading_minutes)} |",
            "",
            "---",
            "",
            "_Generado con el Resumidor de PDF con PLN (TF-IDF, TextRank y MMR)._",
            "",
        ]
    )

    return "\n".join(lines)


def build_json_report(
    document: PDFDocument,
    result: SummaryResult,
    original_stats: TextStats,
    summary_stats: TextStats,
    keywords: list[Keyword],
    language: str,
    source_name: str,
) -> dict[str, object]:
    """Construye un informe JSON serializable."""

    return {
        "documento": {
            "archivo": source_name or document.source_name,
            "titulo": document.title,
            "paginas": document.pages,
            "idioma": language,
            "metadatos": document.metadata,
            "advertencias": list(document.warnings),
        },
        "resumen": {
            "metodo": result.method_label,
            "texto": format_summary(result),
            "oraciones_seleccionadas": list(result.selected),
            "configuracion": result.config,
        },
        "palabras_clave": [asdict(keyword) for keyword in keywords],
        "estadisticas": {
            "original": asdict(original_stats),
            "resumen": asdict(summary_stats),
            "compresion": round(compression_ratio(original_stats, summary_stats), 4),
        },
        "generado": datetime.now().isoformat(timespec="seconds"),
    }


def report_to_json(report: dict[str, object]) -> str:
    """Serializa un informe JSON con acentos legibles."""

    return json.dumps(report, ensure_ascii=False, indent=2)
