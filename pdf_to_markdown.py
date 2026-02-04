#!/usr/bin/env python3
import argparse
import dataclasses
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import fitz
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from vlm_client import VLMClient


@dataclasses.dataclass
class ExtractedImage:
    page_index: int
    path: Path


def render_page_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img.convert("RGB")


def ocr_page(ocr: PaddleOCR, page: fitz.Page) -> str:
    image = render_page_image(page)
    result = ocr.ocr(np.array(image), cls=True)
    if not result or not result[0]:
        return ""
    lines: List[Tuple[float, float, str]] = []
    for entry in result[0]:
        box, (text, _score) = entry
        y = min(point[1] for point in box)
        x = min(point[0] for point in box)
        cleaned = text.strip()
        if cleaned:
            lines.append((y, x, cleaned))
    lines.sort()
    return "\n".join(item[2] for item in lines)


def extract_images(doc: fitz.Document, output_dir: Path) -> List[ExtractedImage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: List[ExtractedImage] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            filename = f"page_{page_index + 1:03d}_img_{img_index:02d}.{ext}"
            image_path = output_dir / filename
            with image_path.open("wb") as f:
                f.write(image_bytes)
            images.append(ExtractedImage(page_index=page_index, path=image_path))
    return images


def group_images_by_page(images: Iterable[ExtractedImage]) -> dict:
    grouped: dict = {}
    for image in images:
        grouped.setdefault(image.page_index, []).append(image)
    return grouped


def build_markdown(
    doc: fitz.Document,
    ocr: PaddleOCR,
    images_by_page: dict,
    vlm_client: Optional[VLMClient],
    output_dir: Path,
) -> str:
    md_lines: List[str] = ["# PDF 转换结果", ""]
    for page_index in range(len(doc)):
        page = doc[page_index]
        md_lines.append(f"## 第 {page_index + 1} 页")
        md_lines.append("")
        page_text = ocr_page(ocr, page)
        if page_text:
            md_lines.append(page_text)
        else:
            md_lines.append("(未检测到文本)")
        md_lines.append("")

        for image in images_by_page.get(page_index, []):
            rel_path = image.path.relative_to(output_dir)
            description = None
            if vlm_client:
                description = vlm_client.describe_image(image.path)
            alt_text = description or "图片"
            md_lines.append(f"![{alt_text}]({rel_path.as_posix()})")
            if description:
                md_lines.append(f"> 图像描述：{description}")
            else:
                md_lines.append("> 图像描述：未提供（未配置 VLM 或调用失败）")
            md_lines.append("")

    return "\n".join(md_lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 PaddleOCR v1.5 将 PDF 转为 Markdown，并为图片生成描述。"
    )
    parser.add_argument("pdf", type=Path, help="PDF 文件路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="输出目录（默认：output）",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="PaddleOCR 语言（默认：ch）",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="跳过调用 VLM 生成图片描述",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    doc = fitz.open(args.pdf)
    images = extract_images(doc, images_dir)
    images_by_page = group_images_by_page(images)

    ocr = PaddleOCR(use_angle_cls=True, lang=args.lang)
    vlm_client = None if args.skip_vlm else VLMClient.from_env()

    markdown = build_markdown(doc, ocr, images_by_page, vlm_client, output_dir)
    md_path = output_dir / f"{args.pdf.stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    print(f"Markdown 已生成：{md_path}")


if __name__ == "__main__":
    main()
