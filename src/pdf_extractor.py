"""Extracción y limpieza de texto desde archivos PDF.

Se usa ``pypdf`` porque es una dependencia ligera, multiplataforma y no
requiere binarios externos. La calidad de extracción depende del PDF: los
documentos escaneados (imágenes) no contienen texto y se detectan para
mostrar un mensaje claro al usuario.
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .errors import (
    PDFEmptyError,
    PDFEncryptedError,
    PDFScannedError,
    PDFTooLargeError,
    UnsupportedFileError,
)

_PAGE_NUMBER_RE = re.compile(r"^[\-–—]?\s*\d{1,4}\s*[\-–—]?$")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[\-–—•*·]\s+")
_NUMBERED_RE = re.compile(r"^\d{1,3}[.)]\s+")
_SYMBOLS_RE = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class PDFDocument:
    """Resultado de la extracción de un PDF."""

    text: str
    pages: int
    page_texts: tuple[str, ...]
    metadata: dict[str, str]
    warnings: tuple[str, ...]
    source_name: str = ""

    @property
    def title(self) -> str:
        return self.metadata.get("title") or Path(self.source_name).stem or "Documento"


def _remove_control_characters(text: str) -> str:
    return "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char)[0] != "C"
    )


def _clean_page_text(raw_text: str) -> str:
    """Limpia una página y reconstruye párrafos."""

    text = unicodedata.normalize("NFKC", raw_text)
    text = _remove_control_characters(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Une palabras partidas con guion al final de línea.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Las URL aportan poco al resumen y ensucian la tokenización.
    text = _URL_RE.sub(" ", text)

    lines = [line.strip() for line in text.split("\n")]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            cleaned.append("")
            continue
        if _PAGE_NUMBER_RE.fullmatch(line):
            continue
        if _SYMBOLS_RE.fullmatch(line):
            continue
        cleaned.append(line)

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer.clear()

    for line in cleaned:
        if not line:
            flush()
            continue
        if _BULLET_RE.match(line) or _NUMBERED_RE.match(line):
            flush()
            paragraphs.append(line)
        else:
            buffer.append(line)
    flush()

    return "\n".join(paragraphs)


def _remove_repeated_lines(page_texts: list[str]) -> tuple[list[str], list[str]]:
    """Elimina encabezados/pies de página repetidos en al menos 50 % de páginas."""

    if len(page_texts) < 4:
        return page_texts, []

    counts: Counter[str] = Counter()
    for page_text in page_texts:
        unique_lines = {line.strip() for line in page_text.split("\n") if line.strip()}
        counts.update(unique_lines)

    threshold = max(3, int(len(page_texts) * 0.5))
    repeated = {
        line
        for line, count in counts.items()
        if count >= threshold and len(line) <= 120
    }
    if not repeated:
        return page_texts, []

    filtered = [
        "\n".join(
            line for line in page_text.split("\n") if line.strip() not in repeated
        )
        for page_text in page_texts
    ]
    return filtered, sorted(repeated)


def extract_text_from_pdf(
    source: bytes | str | Path,
    source_name: str = "",
    max_pages: int = 400,
) -> PDFDocument:
    """Extrae el texto de un PDF y devuelve metadatos y advertencias.

    Parámetros
    ----------
    source:
        Bytes del PDF o ruta al archivo.
    source_name:
        Nombre para mostrar (se infiere si ``source`` es una ruta).
    max_pages:
        Límite de páginas para evitar documentos desproporcionados.

    Raises
    ------
    UnsupportedFileError, PDFEncryptedError, PDFScannedError,
    PDFEmptyError, PDFTooLargeError
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise UnsupportedFileError(f"No se encontró el archivo: {path}")
        data = path.read_bytes()
        source_name = source_name or path.name
    else:
        data = source

    if not data:
        raise UnsupportedFileError("El archivo está vacío.")

    if b"%PDF-" not in data[:1024]:
        raise UnsupportedFileError("El archivo no parece ser un PDF válido.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise UnsupportedFileError(
            "No se pudo leer el PDF. Verifica que el archivo no esté dañado."
        ) from exc
    except Exception as exc:  # pragma: no cover - errores exóticos de pypdf
        raise UnsupportedFileError(f"No se pudo abrir el PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise PDFEncryptedError(
                "El PDF está protegido con contraseña. Retira la protección e inténtalo de nuevo."
            ) from exc
        if not decrypted:
            raise PDFEncryptedError(
                "El PDF está protegido con contraseña. Retira la protección e inténtalo de nuevo."
            )

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise UnsupportedFileError("No se pudieron leer las páginas del PDF.") from exc

    if page_count == 0:
        raise PDFEmptyError("El PDF no contiene páginas.")
    if page_count > max_pages:
        raise PDFTooLargeError(
            f"El PDF tiene {page_count} páginas; el límite es {max_pages}. "
            "Divide el documento o aumenta el límite en el código."
        )

    page_texts: list[str] = []
    empty_pages: list[int] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        cleaned = _clean_page_text(raw_text)
        page_texts.append(cleaned)
        if not cleaned.strip():
            empty_pages.append(page_number)

    page_texts, repeated_lines = _remove_repeated_lines(page_texts)

    if len(empty_pages) == page_count:
        raise PDFScannedError(
            "El PDF no contiene texto seleccionable. Parece un documento escaneado. "
            "Aplica OCR (por ejemplo, con OCRmyPDF) y vuelve a subirlo."
        )

    full_text = "\n\n".join(page_text for page_text in page_texts if page_text.strip())
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    if len(full_text) < 100:
        raise PDFEmptyError(
            "Se extrajo muy poco texto del PDF. Puede estar escaneado, dañado o "
            "contener solo imágenes."
        )

    warnings: list[str] = []
    if empty_pages:
        preview = ", ".join(str(number) for number in empty_pages[:8])
        suffix = "…" if len(empty_pages) > 8 else ""
        warnings.append(
            f"{len(empty_pages)} página(s) sin texto extraíble: {preview}{suffix}. "
            "Es posible que contengan imágenes."
        )
    if repeated_lines:
        warnings.append(
            "Se eliminaron encabezados o pies de página repetidos: "
            + "; ".join(repeated_lines[:3])
            + ("…" if len(repeated_lines) > 3 else "")
        )

    metadata: dict[str, str] = {}
    try:
        raw_metadata = reader.metadata
        if raw_metadata:
            for key in ("title", "author", "subject", "creator"):
                value = getattr(raw_metadata, key, None)
                if value:
                    text_value = str(value).strip()
                    if text_value:
                        metadata[key] = text_value
    except Exception:
        pass

    return PDFDocument(
        text=full_text,
        pages=page_count,
        page_texts=tuple(page_texts),
        metadata=metadata,
        warnings=tuple(warnings),
        source_name=source_name,
    )
