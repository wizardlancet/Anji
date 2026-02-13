"""Markdown AST handling using mistune.

This module provides utilities for parsing Markdown into AST and
rendering AST back to Markdown.
"""

from __future__ import annotations

from typing import Any, Optional
import mistune


class MarkdownAST:
    """A wrapper around mistune's Markdown AST functionality.

    This class provides methods to parse Markdown into an AST (Abstract Syntax Tree)
    and render AST back to Markdown.

    Examples:
        >>> md_ast = MarkdownAST()
        >>> tokens = md_ast.parse("# Hello World")
        >>> rendered = md_ast.render(tokens)
    """

    def __init__(self):
        """Create a MarkdownAST handler."""
        self._md = mistune.create_markdown(renderer=None)

    def parse(self, markdown: str) -> list[dict[str, Any]]:
        """Parse Markdown string into AST.

        Args:
            markdown: The Markdown content to parse.

        Returns:
            A list of AST tokens.

        Example:
            >>> tokens = md_ast.parse("# Heading\\n\\nSome text.")
            >>> tokens[0]
            {'type': 'heading', 'attrs': {'level': 1}, ...}
        """
        return self._md(markdown)

    def find_images(
        self,
        tokens: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find all image tokens in the AST.

        Args:
            tokens: The AST tokens to search.

        Returns:
            A list of image tokens found.

        Example:
            >>> images = md_ast.find_images(tokens)
            >>> for img in images:
            ...     print(img['attrs']['url'])
        """
        images = []
        for token in tokens:
            if token["type"] == "image":
                images.append(token)
            if "children" in token:
                images.extend(self.find_images(token["children"]))
        return images

    def find_headings(
        self,
        tokens: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find all heading tokens in the AST.

        Args:
            tokens: The AST tokens to search.

        Returns:
            A list of heading tokens found, in order.
        """
        headings = []
        for token in tokens:
            if token["type"] == "heading":
                headings.append(token)
            if "children" in token:
                headings.extend(self.find_headings(token["children"]))
        return headings

    def render(
        self,
        tokens: list[dict[str, Any]],
    ) -> str:
        """Render AST tokens back to Markdown.

        Args:
            tokens: The AST tokens to render.

        Returns:
            The rendered Markdown string.

        Example:
            >>> markdown = md_ast.render(tokens)
        """
        return _render_ast_to_markdown(tokens)

    def render_to_file(
        self,
        tokens: list[dict[str, Any]],
        output_path: str,
    ) -> None:
        """Render AST tokens to a Markdown file.

        Args:
            tokens: The AST tokens to render.
            output_path: Path to the output Markdown file.
        """
        from pathlib import Path
        Path(output_path).write_text(self.render(tokens), encoding="utf-8")


def render_inline(tokens: list[dict[str, Any]]) -> str:
    """Render inline tokens to string.

    Args:
        tokens: The inline tokens to render.

    Returns:
        The rendered inline content.
    """
    if not tokens:
        return ""

    parts = []
    for token in tokens:
        t = token["type"]

        if t == "text":
            parts.append(token.get("raw", token.get("text", "")))

        elif t == "strong":
            inner = render_inline(token.get("children", []))
            parts.append(f"**{inner}**")

        elif t == "emphasis":
            inner = render_inline(token.get("children", []))
            parts.append(f"*{inner}*")

        elif t == "codespan":
            raw = token.get("raw", token.get("text", ""))
            parts.append(f"`{raw}`")

        elif t == "link":
            url = token.get("attrs", {}).get("url", "")
            title = token.get("attrs", {}).get("title")
            inner = render_inline(token.get("children", []))
            if title:
                parts.append(f'[{inner}]({url} "{title}")')
            else:
                parts.append(f"[{inner}]({url})")

        elif t == "image":
            attrs = token.get("attrs", {})
            url = attrs.get("url", attrs.get("src", ""))
            alt = render_inline(token.get("children", []))
            desc = attrs.get("desc")
            flowchart = attrs.get("flowchart")
            title = attrs.get("title")

            if title:
                result = f'![{alt}]({url} "{title}")'
            else:
                result = f"![{alt}]({url})"

            if desc:
                result += f"\n\nImage Description: {desc}"
            if flowchart:
                result += f"\n\nFlowchart Code:\n```\n{flowchart}\n```"

            parts.append(result)

        elif t == "linebreak":
            parts.append("  \n")

        elif t == "softbreak":
            parts.append("\n")

        elif t == "inline_html":
            parts.append(token.get("raw", ""))

        elif t == "strikethrough":
            inner = render_inline(token.get("children", []))
            parts.append(f"~~{inner}~~")

        else:
            if "raw" in token:
                parts.append(token["raw"])
            elif "children" in token:
                parts.append(render_inline(token["children"]))

    return "".join(parts)


def _render_block_token(token: dict[str, Any]) -> str:
    """Render a single block token to Markdown string.

    Args:
        token: The block token to render.

    Returns:
        The rendered Markdown string.
    """
    t = token["type"]

    if t == "heading":
        level = token["attrs"]["level"]
        text = render_inline(token.get("children", []))
        return f"{'#' * level} {text}"

    elif t == "paragraph":
        return render_inline(token.get("children", []))

    elif t == "block_code":
        info = token.get("attrs", {}).get("info", "") or ""
        raw = token.get("raw", "").rstrip("\n")
        return f"```{info}\n{raw}\n```"

    elif t == "block_quote":
        inner = _render_ast_to_markdown(token.get("children", []))
        lines = inner.split("\n")
        return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)

    elif t == "list":
        ordered = token.get("attrs", {}).get("ordered", False)
        start = token.get("attrs", {}).get("start", 1)
        items = token.get("children", [])
        parts = []
        for i, item in enumerate(items):
            prefix = f"{start + i}. " if ordered else "- "
            parts.append(_render_list_item(item, prefix))
        return "\n".join(parts)

    elif t == "thematic_break":
        return "---"

    elif t == "block_html":
        return token.get("raw", "").strip()

    elif t == "block_text":
        return render_inline(token.get("children", []))

    elif t == "table":
        return _render_table(token)

    elif t == "blank_line":
        return ""

    else:
        if "raw" in token:
            return token["raw"].strip()
        if "children" in token:
            return render_inline(token["children"])
        return ""


def _render_list_item(token: dict[str, Any], prefix: str) -> str:
    """Render a list item token.

    Args:
        token: The list item token.
        prefix: The list item prefix (e.g., "- " or "1. ").

    Returns:
        The rendered list item.
    """
    children = token.get("children", [])
    parts = []
    for child in children:
        parts.append(_render_block_token(child).strip())
    text = "\n".join(parts)
    lines = text.split("\n")
    indent = " " * len(prefix)
    result = prefix + lines[0]
    for line in lines[1:]:
        result += "\n" + indent + line
    return result


def _render_table(token: dict[str, Any]) -> str:
    """Render a table token.

    Args:
        token: The table token.

    Returns:
        The rendered table.
    """
    if "raw" in token:
        return token["raw"] + "\n"
    return ""


def _render_ast_to_markdown(tokens: list[dict[str, Any]]) -> str:
    """Render a list of AST tokens to a Markdown string.

    Args:
        tokens: The AST tokens to render.

    Returns:
        The rendered Markdown string.
    """
    parts = []
    for token in tokens:
        if token["type"] == "blank_line":
            continue
        rendered = _render_block_token(token).strip()
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)
