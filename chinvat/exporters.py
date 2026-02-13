"""Export AST to various formats.

This module provides utilities for exporting Markdown AST to
different formats including Markdown, JSON, and structured data.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from chinvat.ast_handler import MarkdownAST, render_inline


# Pattern to match markdown image syntax
MD_IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')


def export_to_markdown(
    tokens: list[dict[str, Any]],
    output_path: Optional[str] = None,
    embed_base64: bool = False,
    images_dir: Optional[str] = None,
) -> str:
    """Export AST tokens to Markdown format.

    Args:
        tokens: The AST tokens to export.
        output_path: Optional path to save the output.
        embed_base64: Whether to embed images as base64 data URLs.
        images_dir: Optional directory containing images (used for base64 embedding).

    Returns:
        The Markdown string.
    """
    ast_handler = MarkdownAST()
    markdown = ast_handler.render(tokens)

    # Embed images as base64 if requested
    if embed_base64 and images_dir:
        markdown = embed_images_as_base64(markdown, images_dir)

    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text(markdown, encoding="utf-8")

    return markdown


def embed_images_as_base64(markdown: str, images_dir: str) -> str:
    """Replace image references with base64 data URLs in markdown.

    Args:
        markdown: The markdown content.
        images_dir: Directory containing the images.

    Returns:
        Markdown with images embedded as base64.
    """

    def replace_image(match):
        alt_text = match.group(1)
        image_path = match.group(2)

        # Skip if already base64 or URL
        if image_path.startswith(("http://", "https://", "data:")):
            return match.group(0)

        # Try to find the image file
        # Handle both relative paths (imgs/xxx.jpg) and absolute paths
        if not os.path.isabs(image_path):
            # images_dir is already the full path to the imgs folder
            # But markdown may have paths like "imgs/xxx.jpg", so we need to handle both
            full_path = os.path.join(images_dir, image_path)
            # If not found, try stripping imgs/ prefix
            if not os.path.exists(full_path) and image_path.startswith("imgs/"):
                full_path = os.path.join(images_dir, image_path[5:])  # Remove "imgs/" prefix
        else:
            full_path = image_path

        # Try common extensions if file not found
        if not os.path.exists(full_path):
            base, ext = os.path.splitext(full_path)
            for new_ext in ['.jpg', '.png', '.jpeg', '.gif', '.webp']:
                if os.path.exists(base + new_ext):
                    full_path = base + new_ext
                    break

        if os.path.exists(full_path):
            try:
                suffix = Path(full_path).suffix.lower()
                mime_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".bmp": "image/bmp",
                }
                mime = mime_map.get(suffix, "image/png")

                with open(full_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

                data_url = f"data:{mime};base64,{b64}"
                return f"![{alt_text}]({data_url})"
            except Exception as e:
                print(f"Warning: Failed to embed image {full_path}: {e}")

        return match.group(0)

    return MD_IMG_PATTERN.sub(replace_image, markdown)


def export_to_json(
    tokens: list[dict[str, Any]],
    output_path: Optional[str] = None,
    include_content: bool = True,
) -> dict[str, Any]:
    """Export AST tokens to a structured JSON format.

    Args:
        tokens: The AST tokens to export.
        output_path: Optional path to save the output.
        include_content: Whether to include rendered content strings.

    Returns:
        A dictionary containing the structured data.
    """
    result = {
        "version": "1.0",
        "tokens": _export_tokens(tokens, include_content=include_content),
    }

    if output_path:
        Path(output_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result


def _export_tokens(
    tokens: list[dict[str, Any]],
    include_content: bool = True,
) -> list[dict[str, Any]]:
    """Export tokens to a serializable format.

    Args:
        tokens: The AST tokens.
        include_content: Whether to include rendered content.

    Returns:
        A list of exported token dictionaries.
    """
    result = []
    for token in tokens:
        exported = {
            "type": token["type"],
        }

        # Include attributes
        if "attrs" in token:
            exported["attrs"] = token["attrs"]

        # Include rendered content if requested
        if include_content:
            if token["type"] == "heading":
                exported["content"] = render_inline(token.get("children", []))
            elif token["type"] == "paragraph":
                exported["content"] = render_inline(token.get("children", []))
            elif token["type"] == "image":
                exported["content"] = render_inline(token.get("children", []))
            elif "raw" in token:
                exported["content"] = token["raw"]

        # Include children recursively
        if "children" in token:
            exported["children"] = _export_tokens(token["children"], include_content)

        result.append(exported)

    return result


def export_to_structured_data(
    tokens: list[dict[str, Any]],
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    """Export AST tokens to a structured document format.

    This format is optimized for AI agent consumption with:
    - Flattened headings hierarchy
    - Separated content blocks
    - Image metadata

    Args:
        tokens: The AST tokens.
        output_path: Optional path to save the output.

    Returns:
        A dictionary containing structured document data.
    """
    structure = {
        "title": "",
        "headings": [],
        "content_blocks": [],
        "images": [],
        "tables": [],
    }

    current_section = {"level": 0, "title": "", "content": []}

    for token in tokens:
        if token["type"] == "heading":
            level = token.get("attrs", {}).get("level", 1)
            title = render_inline(token.get("children", []))

            if not structure["title"]:
                structure["title"] = title

            # Save previous section
            if current_section["content"]:
                structure["content_blocks"].append(current_section)

            # Start new section
            current_section = {
                "level": level,
                "title": title,
                "content": [],
            }

            structure["headings"].append({
                "level": level,
                "title": title,
            })

        elif token["type"] == "paragraph":
            content = render_inline(token.get("children", []))
            if content.strip():
                current_section["content"].append({
                    "type": "paragraph",
                    "content": content,
                })

        elif token["type"] == "image":
            attrs = token.get("attrs", {})
            url = attrs.get("url", "")
            caption = render_inline(token.get("children", []))

            image_data = {
                "url": url,
                "caption": caption,
            }
            if "desc" in attrs:
                image_data["description"] = attrs["desc"]
            if "flowchart" in attrs:
                image_data["diagram"] = attrs["flowchart"]
            if "is_informative" in attrs:
                image_data["is_informative"] = attrs["is_informative"]

            structure["images"].append(image_data)
            current_section["content"].append({
                "type": "image",
                "data": image_data,
            })

        elif token["type"] == "list":
            items = _extract_list_items(token)
            current_section["content"].append({
                "type": "list",
                "items": items,
                "ordered": token.get("attrs", {}).get("ordered", False),
            })

        elif token["type"] == "block_code":
            info = token.get("attrs", {}).get("info", "")
            raw = token.get("raw", "")
            current_section["content"].append({
                "type": "code",
                "language": info,
                "content": raw,
            })

        elif token["type"] == "table":
            if "raw" in token:
                structure["tables"].append({
                    "content": token["raw"],
                })

    # Don't forget the last section
    if current_section["content"]:
        structure["content_blocks"].append(current_section)

    if output_path:
        Path(output_path).write_text(
            json.dumps(structure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return structure


def _extract_list_items(token: dict[str, Any]) -> list[str]:
    """Extract text content from list items.

    Args:
        token: The list token.

    Returns:
        A list of item text contents.
    """
    items = []
    for item in token.get("children", []):
        item_text = ""
        for child in item.get("children", []):
            if child["type"] in ("paragraph", "block_text"):
                item_text = render_inline(child.get("children", []))
            elif "raw" in child:
                item_text = child["raw"]
        if item_text:
            items.append(item_text)
    return items


def export_document(
    tokens: list[dict[str, Any]],
    output_path: str,
    format: str = "markdown",
) -> str:
    """Export a document in the specified format.

    Args:
        tokens: The AST tokens.
        output_path: Path to save the output file.
        format: Output format ('markdown', 'json', 'structured').

    Returns:
        The path to the output file.
    """
    output_path = Path(output_path)

    if format == "markdown":
        export_to_markdown(tokens, str(output_path))
    elif format == "json":
        export_to_json(tokens, str(output_path))
    elif format == "structured":
        export_to_structured_data(tokens, str(output_path))
    else:
        raise ValueError(f"Unsupported format: {format}")

    return str(output_path)
