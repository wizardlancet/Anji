<div align="center">

# Chinvat

**PDF to AI Agent Knowledge Bridge**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/chinvat/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/chinvat/chinvat/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/chinvat)](https://pypi.org/project/chinvat/)

Convert PDFs to enhanced, AI-agent-ready Markdown/JSON documents.

[Features](#features) • [Quick Start](#quick-start) • [Installation](#installation) • [Documentation](#documentation)

</div>

---

## What is Chinvat?

Chinvat bridges the gap between PDFs designed for human reading and the structured, semantic text that AI agents require. It leverages:

- **PaddleOCR-VL** for high-quality PDF-to-Markdown conversion
- **Ovis2.5-9B Vision-Language Model** for intelligent image analysis
- **Mistune** for flexible AST manipulation

## Features

| Feature | Description |
|---------|-------------|
| **Smart OCR** | Extracts text, tables, and images with layout awareness |
| **VLM Image Analysis** | Generates captions and descriptions for embedded images |
| **Decorative Filtering** | Removes logos, watermarks, and noise automatically |
| **Heading Correction**(developing) | Fixes OCR-generated heading hierarchy issues |
| **Multi-Format Output** | Export to Markdown, JSON, or structured data |
| **Batch Processing** | Efficiently process multiple PDFs |
| **Flexible Pipeline** | Run full pipeline or individual steps |

## Quick Start

```bash
# Install
pip install -e .

# Convert a PDF
chinvat pipeline document.pdf output/

# Or use as a Python library
python -c "
from chinvat import run_full_pipeline
run_full_pipeline('document.pdf', 'output/')
"
```

## Installation

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Prerequisites

Chinvat requires two external services:

1. **PaddleOCR-VL Server** (port 8118)
2. **Ovis2.5-9B VLM Server** (port 8000)

See [references/readme.md](references/readme.md) for setup instructions.

## Usage

### Command Line

```bash
# Full pipeline
chinvat pipeline input.pdf output_dir

# Batch processing
chinvat batch output_base file1.pdf file2.pdf file3.pdf

# Individual steps
chinvat pdf input.pdf output_dir          # PDF → Markdown
chinvat image input.md output.md          # Analyze images
chinvat md enhance input.md output.md     # Enhance AST
chinvat md export input.md out --format json  # Export
```

### Python API

```python
from chinvat import Pipeline, run_full_pipeline, batch_pipeline

# Simple usage
run_full_pipeline("document.pdf", "output/")

# Advanced usage
pipeline = Pipeline(
    paddleocr_server_url="http://localhost:8118/v1",
    vlm_server_url="http://localhost:8000/v1"
)

outputs = pipeline.run(
    input_path="document.pdf",
    output_folder="output",
    format="both"  # markdown, json, structured, or both
)

# Batch processing
batch_pipeline(
    input_paths=["doc1.pdf", "doc2.pdf"],
    output_base_folder="batch_output"
)

pipeline.close()
```

## Output Structure

```
output/
└── document_name/
    └── enhanced/
        ├── document.md     # Enhanced Markdown
        └── document.json   # JSON AST (optional)
```

## Architecture

```
PDF Document
     │
     ▼
┌─────────────────┐
│ 1. PDF → Markdown │  PaddleOCR-VL
│    (pdf_converter.py) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Markdown → AST │  Mistune
│     (ast_handler.py)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Enhance AST   │  Image Analysis + Filtering
│ (enhancement.py) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Export        │  Markdown / JSON / Structured
│   (exporters.py) │
└─────────────────┘
         │
         ▼
   AI-Ready Output
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000/v1` | VLM server URL |
| `API_KEY` | `abc-123` | VLM API key |
| `MODEL_NAME` | `AIDC-AI/Ovis2.5-9B` | VLM model name |

### CLI Options

```bash
chinvat pipeline input.pdf output/ \
  --format markdown|json|structured|both \
  --no-enhance \
  --no-fix-headings \
  --no-filter-decorative \
  --no-enrich-images \
  --dummy  # Test without API calls
```

## Development

```bash
# Code formatting
black chinvat/

# Linting
ruff check chinvat/

# Type checking
mypy chinvat/

# Testing
pytest
```

## Project Structure

```
chinvat/
├── chinvat/              # Main package
│   ├── __init__.py       # Exports
│   ├── main.py           # CLI entry point
│   ├── cli.py            # Command-line interface
│   ├── pipeline.py       # Pipeline orchestration
│   ├── pdf_converter.py  # PDF → Markdown
│   ├── image_analyzer.py # VLM image analysis
│   ├── ast_handler.py    # AST manipulation
│   ├── enhancement.py    # AST enhancement
│   └── exporters.py     # Export utilities
├── pyproject.toml        # Package configuration
├── README.md             # English documentation
├── README_CN.md          # 中文文档
├── CLAUDE.md             # Claude Code context
├── .gitignore
└── references/           # Reference implementations
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read [CLAUDE.md](CLAUDE.md) for development guidelines.

---

<div align="center">

**Built for AI agents, by AI agents**

</div>
