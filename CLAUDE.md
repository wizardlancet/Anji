# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Anji-Bridge (安济桥)** is a Python package that converts PDFs to AI-agent-ready Markdown/JSON documents. It uses PaddleOCR-VL for PDF-to-Markdown conversion and Ovis2.5-9B (VLM) for image analysis, with Mistune for AST manipulation.

## Project Structure

```
anji/                   # GitHub repository root
├── anji/             # Main Python package
│   ├── __init__.py   # Package exports
│   ├── main.py       # CLI entry point
│   ├── __main__.py   # python -m anji
│   ├── cli.py        # CLI commands
│   ├── pipeline.py   # Pipeline orchestration
│   ├── pdf_converter.py
│   ├── image_analyzer.py
│   ├── ast_handler.py
│   ├── enhancement.py
│   └── exporters.py
├── pyproject.toml     # Package config
├── README.md         # English docs
├── README_CN.md      # Chinese docs
├── CLAUDE.md         # AI context
└── references/       # Reference implementations
```

## Development Commands

```bash
# Install package in development mode
pip install -e .

# Run tests
pytest

# Run a specific test file
pytest tests/test_pipeline.py

# Format code
black anji/

# Lint code
ruff check anji/

# Type checking
mypy anji/
```

## Architecture

**4-Stage Processing Pipeline** (`anji/pipeline.py`):
1. **PDF → Raw Markdown** (`pdf_converter.py`) - PaddleOCR-VL server on port 8118
2. **Markdown → AST** (`ast_handler.py`) - Parse with Mistune
3. **AST → Enhanced AST** (`enhancement.py`, `image_analyzer.py`) - VLM image analysis, heading fixes, decorative filtering
4. **AST → Output** (`exporters.py`) - Export to Markdown, JSON, or structured data

**Module Responsibilities:**
- `cli.py` - Full CLI with subcommands: pipeline, batch, pdf, image, md
- `pdf_converter.py` - PDFToMarkdownConverter (lazy singleton for batch efficiency)
- `image_analyzer.py` - Async VLM calls with semaphore concurrency control (max 3)
- `ast_handler.py` - MarkdownAST wrapper around Mistune
- `enhancement.py` - Heading fixes, image filtering, enrichment
- `exporters.py` - export_to_markdown, export_to_json, export_to_structured_data

**Exported API** (`anji/__init__.py`):
```python
from anji import Pipeline, run_full_pipeline, batch_pipeline
from anji import PDFToMarkdownConverter, MarkdownAST, ImageAnalyzer, Enhancer
from anji import export_to_markdown, export_to_json
```

## External Service Dependencies

- **PaddleOCR-VL**: `http://localhost:8118/v1` (vllm-server backend)
- **Ovis2.5-9B VLM**: `http://localhost:8000/v1` with API key `abc-123`

Environment variables: `API_BASE_URL`, `API_KEY`, `MODEL_NAME`, `PROMPT`

## CLI Usage

```bash
# Full pipeline
anji pipeline input.pdf output_dir [--format markdown|json|structured|both]

# Batch processing
anji batch output_base file1.pdf [file2.pdf...]

# Individual steps
anji pdf input.pdf output_dir [--no-merge-tables]
anji image input.md output.md [--vlm-url http://localhost:8000/v1]
anji md enhance input.md output.md
anji md export input.md output [--format markdown|json|structured]

# All modules support --dummy flag for testing without API calls

# Run as Python module
python -m anji pipeline input.pdf output_dir
```

## Code Style

- Line length: 100
- Python: 3.10+
- Formatter: Black
- Linter: Ruff
- Type checker: Mypy (strict, disallow_untyped_defs)
