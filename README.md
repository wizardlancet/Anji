<div align="center">

# Chinvat

**PDF to AI Agent Knowledge Bridge**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/chinvat/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/chinvat/chinvat/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/chinvat)](https://pypi.org/project/chinvat/)

Convert PDFs to enhanced, AI-agent-ready Markdown/JSON documents.

[Features](#features) • [Quick Start](#quick-start) • [Installation](#installation) • [Usage](#usage)

</div>

---

[English](./README.md) | [中文](./README_CN.md)

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
| **Base64 Embedding** | Embed images as base64 data URLs in markdown |

## Quick Start

```bash
# Install
pip install -e .

# Convert a PDF
chinvat pipeline document.pdf output/

# Embed images as base64 (single portable file)
chinvat pipeline document.pdf output/ --embed-base64

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

Chinvat requires two external services running:

#### 1. PaddleOCR-VL Server (port 8118)

Requires GPU. Run using Docker:

```bash
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
    paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

#### 2. Ovis2.5-9B VLM Server (port 8000)

Requires GPU with ~16GB VRAM. Run with vLLM:

```bash
vllm serve AIDC-AI/Ovis2.5-9B \
    --trust-remote-code \
    --port 8000 \
    --gpu-memory-utilization 0.4
```

> **Note:** If you encounter `RuntimeError: Exception from the 'vlm' worker: only 0-dimensional arrays can be converted to Python scalars`, install `numpy==1.26.4`.

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

### Output Options

```bash
# Keep images folder (default: enabled)
chinvat pipeline input.pdf output/ --keep-images

# Disable images folder
chinvat pipeline input.pdf output/ --no-keep-images

# Embed images as base64 (single portable markdown file)
chinvat pipeline input.pdf output/ --embed-base64

# Combine options
chinvat pipeline input.pdf output/ --embed-base64 --no-keep-images
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
    output_format="both",  # markdown, json, structured, or both
    keep_images=True,  # keep imgs folder
    embed_base64=False,  # or True for single file
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
        ├── document.json   # JSON AST (optional)
        └── imgs/          # Extracted images (optional)
            ├── image1.jpg
            └── image2.jpg
```

With `--embed-base64`, images are embedded directly in the markdown file as base64 data URLs.

## How It Works

Chinvat processes PDFs through 4 stages:

1. **PDF → Markdown** - Uses PaddleOCR-VL to extract text, tables, and images
2. **Markdown → AST** - Parses markdown into an abstract syntax tree using Mistune
3. **Enhance** - Analyzes images with VLM, fixes heading levels, filters decorative elements
4. **Export** - Outputs as Markdown, JSON, or structured data

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
  --keep-images \
  --embed-base64 \
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
│   └── exporters.py      # Export utilities
├── pyproject.toml        # Package configuration
├── README.md             # English documentation
├── README_CN.md          # Chinese documentation
├── CLAUDE.md             # Claude Code context
└── .gitignore
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read [CLAUDE.md](CLAUDE.md) for development guidelines.

---

<div align="center">

**Built for AI agents, by AI agents**

</div>
