"""Anji (安济桥) - PDF to AI Agent Knowledge Bridge

A Python package for converting PDFs to enhanced Markdown/JSON,
optimized for AI agent consumption.
"""

__version__ = "0.1.0"

from anji.pipeline import Pipeline, run_full_pipeline, batch_pipeline
from anji.pdf_converter import PDFToMarkdownConverter, get_default_converter
from anji.ast_handler import MarkdownAST
from anji.image_analyzer import ImageAnalyzer, get_default_analyzer
from anji.enhancement import Enhancer
from anji.exporters import export_to_markdown, export_to_json

__all__ = [
    # Pipeline
    "Pipeline",
    "run_full_pipeline",
    "batch_pipeline",
    # PDF Converter
    "PDFToMarkdownConverter",
    "get_default_converter",
    # AST Handler
    "MarkdownAST",
    # Image Analyzer
    "ImageAnalyzer",
    "get_default_analyzer",
    # Enhancement
    "Enhancer",
    # Exporters
    "export_to_markdown",
    "export_to_json",
]
