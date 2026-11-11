#!/usr/bin/env python3
"""Interfaz de línea de comandos del resumidor de PDF.

Ejemplos:
    python cli.py documento.pdf
    python cli.py documento.pdf -m textrank -r 0.15 --keywords 10
    python cli.py documento.pdf --json -o analisis.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.errors import PDFSummarizerError
from src.keywords import extract_keywords
from src.metrics import compression_ratio, compute_stats
from src.pdf_extractor import extract_text_from_pdf
from src.preprocessing import detect_language, preprocess_text
from src.summarizers import METHOD_LABELS, summarize
from src.utils import (
    build_json_report,
    format_summary,
    reading_time_label,
    report_to_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume un PDF con TF-IDF, TextRank y MMR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", type=Path, help="Ruta al archivo PDF.")
    parser.add_argument(
        "-m",
        "--metodo",
        choices=list(METHOD_LABELS.keys()),
        default="tfidf",
        help="Algoritmo de resumen.",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.2,
        help="Proporción de oraciones a conservar (0.1 a 0.5).",
    )
    parser.add_argument(
        "--sin-mmr",
        action="store_true",
        help="Desactiva la selección con diversidad (MMR).",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="Balance relevancia/diversidad de MMR (0 a 1).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=6,
        help="Mínimo de palabras por oración.",
    )
    parser.add_argument(
        "--idioma",
        choices=["auto", "es", "en"],
        default="auto",
        help="Idioma del documento.",
    )
    parser.add_argument(
        "--keywords",
        type=int,
        default=0,
        metavar="N",
        help="Muestra las N palabras clave principales (0 = no mostrar).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Muestra estadísticas del documento y del resumen.",
    )
    parser.add_argument(
        "--vinetas",
        action="store_true",
        help="Presenta el resumen como lista de viñetas.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emite un informe JSON completo.",
    )
    parser.add_argument(
        "-o",
        "--salida",
        type=Path,
        help="Archivo de salida. Si se omite, se imprime en stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 0.0 < args.ratio <= 1.0:
        print("Error: --ratio debe estar entre 0 y 1.", file=sys.stderr)
        return 2
    if not 0.0 <= args.mmr_lambda <= 1.0:
        print("Error: --mmr-lambda debe estar entre 0 y 1.", file=sys.stderr)
        return 2

    try:
        document = extract_text_from_pdf(args.pdf)
    except PDFSummarizerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.idioma == "auto":
        language = detect_language(document.text)
    else:
        language = "spanish" if args.idioma == "es" else "english"

    sentences = preprocess_text(
        document.text,
        language=language,
        min_words=args.min_words,
    )
    if not sentences:
        print(
            "Error: no se encontraron oraciones suficientes para resumir.",
            file=sys.stderr,
        )
        return 2

    result = summarize(
        sentences,
        method=args.metodo,
        ratio=args.ratio,
        use_mmr=not args.sin_mmr,
        mmr_lambda=args.mmr_lambda,
    )
    keyword_count = args.keywords or (15 if args.json else 0)
    keywords = extract_keywords(sentences, top_n=keyword_count) if keyword_count else []
    original_stats = compute_stats(document.text)
    summary_stats = compute_stats(result.summary)

    if args.json:
        report = build_json_report(
            document,
            result,
            original_stats,
            summary_stats,
            keywords,
            language,
            str(args.pdf),
        )
        output = report_to_json(report)
    else:
        sections: list[str] = [format_summary(result, bullets=args.vinetas)]

        if args.keywords and keywords:
            sections.append("\nPalabras clave:")
            sections.extend(
                f"  - {keyword.term} ({keyword.score:.2f})"
                for keyword in keywords
            )

        if args.stats:
            sections.extend(
                [
                    "\nEstadísticas:",
                    f"  Documento: {original_stats.words:,} palabras, "
                    f"{original_stats.sentences:,} oraciones, "
                    f"{reading_time_label(original_stats.reading_minutes)} de lectura.",
                    f"  Resumen:   {summary_stats.words:,} palabras, "
                    f"{summary_stats.sentences:,} oraciones, "
                    f"{reading_time_label(summary_stats.reading_minutes)} de lectura.",
                    f"  Compresión: {compression_ratio(original_stats, summary_stats):.0%}.",
                ]
            )

        output = "\n".join(sections)

    if args.salida:
        args.salida.write_text(output, encoding="utf-8")
        print(f"Resultado guardado en {args.salida}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
