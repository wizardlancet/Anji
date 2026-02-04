# PDF to Markdown (PaddleOCR v1.5)

该工具基于 **PaddleOCR v1.5** 完成以下流程：

1. 将 PDF 转为 Markdown。
2. 抽取 PDF 中的图片，保存为独立文件，并在 Markdown 中嵌入。
3. 调用 Vision Language Model (VLM) 为图片生成描述，并补充进 Markdown。

## 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：PaddleOCR 依赖 `paddlepaddle`，请根据你的平台安装对应版本。

## 使用方法

```bash
python pdf_to_markdown.py /path/to/file.pdf --output-dir output
```

默认会读取环境变量配置 VLM 调用：

- `VLM_API_KEY`：API Key（必填，否则跳过 VLM）
- `VLM_ENDPOINT`：API 端点（默认 OpenAI 兼容接口）
- `VLM_MODEL`：模型名称（默认 `gpt-4o-mini`）
- `VLM_PROMPT`：描述提示词

如需跳过 VLM：

```bash
python pdf_to_markdown.py /path/to/file.pdf --skip-vlm
```

输出结构示例：

```
output/
  file.md
  images/
    page_001_img_01.png
    page_001_img_02.png
```

## 注意事项

- OCR 识别的文本会按坐标排序，可能需要进一步后处理。
- 如果 PDF 中没有嵌入图片，Markdown 中不会生成图片段落。
