<div align="center">

# Anji-Bridge (安济桥)

**PDF 转 AI Agent 知识桥梁**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/anji/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/wizardlancet/Anji/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/anji)](https://pypi.org/project/anji/)

将 PDF 转换为增强的、AI Agent 可读的 Markdown/JSON 文档。

[功能特性](#功能特性) • [快速开始](#快速开始) • [安装](#安装) • [使用文档](#使用文档)

</div>

---

[English](./README.md) | [中文](./README_CN.md)

---

## 什么是 Anji？

Anji 希望解决 PDF 格式 与 AI Agent 之间的鸿沟。PDF 是为人类阅读设计的固定版面格式，而 AI Agent 需要的是结构化、语义明确的文本。

Anji 采用以下技术栈：

- **PaddleOCR-VL** - 高质量 PDF 转 Markdown
- **Ovis2.5-9B 视觉语言模型** - 智能图像分析
- **Mistune** - 灵活的 AST 操作

## 功能特性

| 功能 | 描述 |
|------|------|
| **智能 OCR** | 提取文本、表格和图片，保持版面结构 |
| **VLM 图像分析** | 为嵌入图片生成标题和描述 |
| **装饰元素过滤** | 自动移除 logo、水印、分割线等 |
| **标题层级修正**(developing) | 修复 OCR 输出的标题层级问题 |
| **多格式输出** | 支持 Markdown、JSON、Structured 格式 |
| **批量处理** | 高效处理多个 PDF 文件 |
| **灵活流水线** | 可运行完整流程或单独步骤 |
| **Base64 嵌入** | 将图片嵌入为 base64 数据 URL，单文件便携 |

## 快速开始

```bash
# 安装
pip install -e .

# 转换 PDF
anji pipeline document.pdf output/

# 嵌入图片为 base64（单个便携文件）
anji pipeline document.pdf output/ --embed-base64

# 或作为 Python 库使用
python -c "
from anji import run_full_pipeline
run_full_pipeline('document.pdf', 'output/')
"
```

## 安装

```bash
# 基础安装
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"
```

### 前置条件

Anji 需要运行两个外部服务：

#### 1. PaddleOCR-VL 服务器（端口 8118）

需要 GPU。使用 Docker 运行：

```bash
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
    paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

#### 2. Ovis2.5-9B VLM 服务器（端口 8000）

需要 GPU，约 16GB 显存。使用 vLLM 运行：

```bash
vllm serve AIDC-AI/Ovis2.5-9B \
    --trust-remote-code \
    --port 8000 \
    --gpu-memory-utilization 0.4
```

> **注意：** 如果遇到 `RuntimeError: Exception from the 'vlm' worker: only 0-dimensional arrays can be converted to Python scalars`，请安装 `numpy==1.26.4`。

## 使用文档

### 命令行

```bash
# 完整流水线
anji pipeline input.pdf output_dir

# 批量处理
anji batch output_base file1.pdf file2.pdf file3.pdf

# 单独步骤
anji pdf input.pdf output_dir          # PDF → Markdown
anji image input.md output.md          # 分析图片
anji md enhance input.md output.md     # 增强 AST
anji md export input.md out --format json  # 导出
```

### 输出选项

```bash
# 保留图片文件夹（默认：启用）
anji pipeline input.pdf output/ --keep-images

# 不保留图片文件夹
anji pipeline input.pdf output/ --no-keep-images

# 嵌入图片为 base64（单个便携 markdown 文件）
anji pipeline input.pdf output/ --embed-base64

# 组合使用
anji pipeline input.pdf output/ --embed-base64 --no-keep-images
```

### Python API

```python
from anji import Pipeline, run_full_pipeline, batch_pipeline

# 简单用法
run_full_pipeline("document.pdf", "output/")

# 高级用法
pipeline = Pipeline(
    paddleocr_server_url="http://localhost:8118/v1",
    vlm_server_url="http://localhost:8000/v1"
)

outputs = pipeline.run(
    input_path="document.pdf",
    output_folder="output",
    output_format="both",  # markdown, json, structured, 或 both
    keep_images=True,  # 保留 imgs 文件夹
    embed_base64=False,  # 设为 True 生成单文件
)

# 批量处理
batch_pipeline(
    input_paths=["doc1.pdf", "doc2.pdf"],
    output_base_folder="batch_output"
)

pipeline.close()
```

## 输出结构

```
output/
└── document_name/
    └── enhanced/
        ├── document.md     # 增强后的 Markdown
        ├── document.json   # JSON AST（可选）
        └── imgs/          # 提取的图片（可选）
            ├── image1.jpg
            └── image2.jpg
```

使用 `--embed-base64` 时，图片将直接以 base64 数据 URL 形式嵌入到 markdown 文件中。

## 工作原理

Anji 通过 4 个阶段处理 PDF：

1. **PDF → Markdown** - 使用 PaddleOCR-VL 提取文本、表格和图片
2. **Markdown → AST** - 使用 Mistune 将 markdown 解析为抽象语法树
3. **增强** - 使用 VLM 分析图片，修正标题层级，过滤装饰元素
4. **导出** - 输出为 Markdown、JSON 或结构化数据

## 配置说明

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `API_BASE_URL` | `http://localhost:8000/v1` | VLM 服务器地址 |
| `API_KEY` | `abc-123` | VLM API 密钥 |
| `MODEL_NAME` | `AIDC-AI/Ovis2.5-9B` | VLM 模型名称 |

### CLI 选项

```bash
anji pipeline input.pdf output/ \
  --format markdown|json|structured|both \
  --keep-images \           # 保留图片文件夹（默认）
  --no-keep-images \        # 不保留图片文件夹
  --embed-base64 \          # 嵌入图片为 base64
  --no-enhance \
  --no-fix-headings \
  --no-filter-decorative \
  --no-enrich-images \
  --dummy  # 测试时不调用 API
```

## 开发指南

```bash
# 代码格式化
black anji/

# 代码检查
ruff check anji/

# 类型检查
mypy anji/

# 运行测试
pytest
```

## 项目结构

```
anji/
├── anji/              # 主包
│   ├── __init__.py       # 导出接口
│   ├── main.py           # CLI 入口
│   ├── cli.py            # 命令行接口
│   ├── pipeline.py       # 流水线编排
│   ├── pdf_converter.py  # PDF 转 Markdown
│   ├── image_analyzer.py # VLM 图像分析
│   ├── ast_handler.py    # AST 操作
│   ├── enhancement.py    # AST 增强
│   └── exporters.py      # 导出工具
├── pyproject.toml        # 包配置
├── README.md            # 英文文档
├── README_CN.md         # 中文文档
├── CLAUDE.md            # Claude Code 上下文
└── .gitignore
```

## 许可证

MIT License。详见 [LICENSE](LICENSE)。

## 贡献指南

欢迎贡献代码！开发前请阅读 [CLAUDE.md](CLAUDE.md) 了解开发规范。

---

<div align="center">

**为 AI Agent 而生，由 AI Agent 构建**

</div>
