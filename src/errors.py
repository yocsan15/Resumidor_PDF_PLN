"""Excepciones propias del resumidor de PDF.

Tener errores específicos permite que la interfaz web y la CLI muestren
mensajes claros para cada situación (PDF cifrado, escaneado, vacío, etc.)
en lugar de un mensaje genérico de Python.
"""

from __future__ import annotations


class PDFSummarizerError(Exception):
    """Error base del dominio del resumidor."""


class UnsupportedFileError(PDFSummarizerError):
    """El archivo no es un PDF válido."""


class PDFEncryptedError(PDFSummarizerError):
    """El PDF está protegido con contraseña."""


class PDFScannedError(PDFSummarizerError):
    """El PDF no contiene texto seleccionable (probablemente escaneado)."""


class PDFEmptyError(PDFSummarizerError):
    """El PDF no contiene texto suficiente para procesar."""


class PDFTooLargeError(PDFSummarizerError):
    """El PDF supera el límite de páginas permitido."""


class NotEnoughTextError(PDFSummarizerError):
    """El texto no contiene oraciones suficientes para resumir."""
