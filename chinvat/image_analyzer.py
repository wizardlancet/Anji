"""Image analysis using Ovis2.5-9B model.

This module provides utilities for analyzing images in Markdown documents
using the Ovis2.5-9B visual language model.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from openai import AsyncOpenAI


# ========== Configuration ==========
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("API_KEY", "abc-123")
MODEL_NAME = os.getenv("MODEL_NAME", "AIDC-AI/Ovis2.5-9B")

DEFAULT_PROMPT = """Describe the content of this image and return the result in JSON format:

{"caption": "A brief title", "description": "A detailed description", "diagram": "Graphviz or Mermaid code", "is_informative": true}

## Language Rules
- **Default language**: English.
- **Exception**: If the image contains visible text, respond in the **same language** as the text found in the image. If multiple languages are present, use the **dominant** one.
- **Important**: Language rules apply **only** to the JSON values. The JSON keys (`caption`, `description`, `diagram`, `is_informative`) must **always** remain in English, regardless of the response language.

## Guidelines
- `caption`: A concise title summarizing the main subject (under 15 words).
- `description`: A detailed description covering key elements such as objects, people, actions, colors, composition, background, mood, and any visible text content.
- `diagram`: If the image contains a flowchart, process diagram, state diagram, or any structured diagram, reproduce it as accurately as possible using **Mermaid** (preferred) or **Graphviz DOT** syntax. Preserve all node labels, edges, directions, and logical relationships from the original image. If the image does not contain any diagram, set this field to `null`.
- `is_informative`: A boolean indicating whether the image carries meaningful semantic content for the document. Set to `true` for images that convey substantive information (e.g. charts, screenshots, photographs, diagrams, tables). Set to `false` for decorative or low-semantic elements (e.g. logos, icons, dividers, background patterns, watermarks, QR code).
- Output **only** the raw JSON object. Do not include any markdown formatting, code fences, or extra commentary."""

PROMPT = os.getenv("PROMPT", DEFAULT_PROMPT)

MAX_CONCURRENCY = 3
MAX_RETRIES = 3

# ========== Dummy mode return values ==========
DUMMY_RESPONSE = {
    "caption": "test caption",
    "description": "test description",
    "diagram": None,
    "is_informative": True,
}

# ========== Patterns ==========
MD_IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
HTML_IMG_PATTERN = re.compile(r'<img\s+[^>]*?src=["\']([^"\']+)["\'][^>]*?/?>', re.IGNORECASE)


class ImageAnalyzer:
    """An image analyzer using Ovis2.5-9B model.

    This class provides methods to analyze images and generate captions,
    descriptions, and diagram representations.

    Examples:
        >>> analyzer = ImageAnalyzer()
        >>> result = analyzer.analyze("path/to/image.png")
        >>> print(result["caption"])
    """

    _instance: Optional["ImageAnalyzer"] = None
    _client: Optional[AsyncOpenAI] = None

    def __init__(
        self,
        api_base_url: str = API_BASE_URL,
        api_key: str = API_KEY,
        model_name: str = MODEL_NAME,
        prompt: str = DEFAULT_PROMPT,
        max_concurrency: int = MAX_CONCURRENCY,
        dummy_mode: bool = False,
    ):
        """Create an image analyzer.

        Args:
            api_base_url: API base URL. Defaults to "http://127.0.0.1:8000/v1".
            api_key: API key. Defaults to "abc-123".
            model_name: Model name. Defaults to "AIDC-AI/Ovis2.5-9B".
            prompt: Prompt for the model.
            max_concurrency: Maximum concurrent API calls.
            dummy_mode: If True, return dummy responses without calling API.
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        self.prompt = prompt
        self.max_concurrency = max_concurrency
        self.dummy_mode = dummy_mode
        self._client = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
            )
        return self._client

    def close(self) -> None:
        """Close the client and release resources."""
        if self._client is not None:
            self._client = None

    @classmethod
    def get_default(
        cls,
        api_base_url: str = API_BASE_URL,
        api_key: str = API_KEY,
        model_name: str = MODEL_NAME,
    ) -> "ImageAnalyzer":
        """Get the default analyzer (singleton).

        Args:
            api_base_url: API base URL.
            api_key: API key.
            model_name: Model name.

        Returns:
            An ImageAnalyzer instance.
        """
        if cls._instance is None or cls._instance.api_base_url != api_base_url:
            cls._instance = cls(
                api_base_url=api_base_url,
                api_key=api_key,
                model_name=model_name,
            )
        return cls._instance

    def analyze_image(self, image_path: Path) -> Optional[dict]:
        """Analyze a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            A dictionary with caption, description, diagram, and is_informative,
            or None if analysis failed.
        """
        if self.dummy_mode:
            return DUMMY_RESPONSE.copy()

        data_url = self._encode_image(image_path)
        if not data_url:
            return None

        result = asyncio.run(self._call_vlm(data_url))
        return result

    async def analyze_images(
        self,
        image_paths: list[Path],
        save_json: bool = True,
    ) -> dict[str, dict]:
        """Analyze multiple images concurrently.

        Args:
            image_paths: List of image paths to analyze.
            save_json: Whether to save JSON results alongside images.

        Returns:
            A dictionary mapping image paths to their analysis results.
        """
        if self.dummy_mode:
            return {str(p): DUMMY_RESPONSE.copy() for p in image_paths}

        # Create client locally to avoid cross-event-loop issues
        # when asyncio.run() is called multiple times
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )

        try:
            sem = asyncio.Semaphore(self.max_concurrency)
            tasks = [
                self._process_image(sem, p, save_json=save_json, client=client)
                for p in image_paths
            ]
            results = await asyncio.gather(*tasks)
        finally:
            await client.close()

        return {path: data for r in results if r for path, data in [r]}

    async def _process_image(
        self,
        sem: asyncio.Semaphore,
        image_path: Path,
        save_json: bool = True,
        client: Optional[AsyncOpenAI] = None,
    ) -> Optional[tuple[str, dict]]:
        """Process a single image with concurrency control.

        Args:
            sem: Semaphore for concurrency control.
            image_path: Path to the image.
            save_json: Whether to save JSON results.
            client: AsyncOpenAI client to use for API calls.

        Returns:
            Tuple of (image path string, analysis result), or None.
        """
        async with sem:
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                return None

            data_url = self._encode_image(image_path)
            if not data_url:
                return None

            result = await self._call_vlm(data_url, client=client)
            if not result:
                return None

            if save_json:
                json_path = image_path.with_suffix(".json")
                try:
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to save JSON for {image_path}: {e}")

            return (str(image_path), result)

    async def _call_vlm(
        self,
        image_data_url: str,
        enable_thinking: bool = False,
        max_tokens: int = 2048,
        client: Optional[AsyncOpenAI] = None,
    ) -> Optional[dict]:
        """Call the VLM model to analyze an image.

        Args:
            image_data_url: The image data URL.
            enable_thinking: Whether to enable thinking mode.
            max_tokens: Maximum number of tokens for the response.
            client: AsyncOpenAI client to use for API calls.

        Returns:
            Analysis result dictionary, or None on failure.
        """
        # Use provided client or fall back to cached client
        api_client = client or self.client

        for attempt in range(MAX_RETRIES):
            try:
                resp = await api_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                            ],
                        }
                    ],
                    max_tokens=max_tokens,
                    extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    },
                },
                )
                content = resp.choices[0].message.content
                result = self._extract_json(content)
                if result:
                    return result
                return {"caption": content.strip()[:100], "description": content.strip()}
            except Exception as e:
                print(f"Warning: API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

    def _encode_image(self, image_path: Path) -> Optional[str]:
        """Encode an image as base64.

        Args:
            image_path: Path to the image.

        Returns:
            The base64-encoded image data URL, or None if failed.
        """
        if not image_path.exists():
            return None

        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(suffix, "image/png")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        return f"data:{mime};base64,{b64}"

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from text.

        Args:
            text: The text containing JSON.

        Returns:
            The parsed JSON dictionary, or None.
        """
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None


def get_default_analyzer(
    api_base_url: str = API_BASE_URL,
    api_key: str = API_KEY,
    model_name: str = MODEL_NAME,
) -> ImageAnalyzer:
    """Get the default image analyzer.

    Args:
        api_base_url: API base URL.
        api_key: API key.
        model_name: Model name.

    Returns:
        An ImageAnalyzer instance.
    """
    return ImageAnalyzer.get_default(
        api_base_url=api_base_url,
        api_key=api_key,
        model_name=model_name,
    )


def update_markdown_images(
    content: str,
    results: dict[str, dict],
) -> str:
    """Update Markdown image references with captions.

    Args:
        content: The original Markdown content.
        results: Dictionary mapping image paths to analysis results.

    Returns:
        The updated Markdown content.
    """
    norm_results = {normalize_path(k): v for k, v in results.items()}

    def replace_md_img(m):
        alt, path = m.group(1), m.group(2)
        norm_path = normalize_path(path)
        if norm_path in norm_results and "caption" in norm_results[norm_path]:
            caption = norm_results[norm_path]["caption"].replace("]", "\\]")
            return f"![{caption}]({path})"
        return m.group(0)

    def replace_html_img(m):
        full_match = m.group(0)
        src = m.group(1)
        norm_src = normalize_path(src)
        if norm_src in norm_results and "caption" in norm_results[norm_src]:
            caption = norm_results[norm_src]["caption"].replace('"', "&quot;")
            if re.search(r'\balt=["\']', full_match, re.IGNORECASE):
                return re.sub(r'\balt=["\'][^"\']*["\']', f'alt="{caption}"', full_match)
            else:
                return full_match.replace("<img ", f'<img alt="{caption}" ', 1)
        return full_match

    content = MD_IMG_PATTERN.sub(replace_md_img, content)
    content = HTML_IMG_PATTERN.sub(replace_html_img, content)
    return content


def normalize_path(path: str) -> str:
    """Normalize a path string.

    Args:
        path: The path to normalize.

    Returns:
        The normalized path.
    """
    return unquote(path).lstrip("./")


def find_markdown_images(content: str) -> list[str]:
    """Find all local image references in Markdown.

    Args:
        content: The Markdown content.

    Returns:
        A list of unique image paths.
    """
    md_images = MD_IMG_PATTERN.findall(content)
    html_images = HTML_IMG_PATTERN.findall(content)
    all_paths = list(set([p for _, p in md_images] + html_images))

    # Filter URLs (only process local files)
    return [p for p in all_paths if not p.startswith(("http://", "https://", "data:"))]


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for image analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze images in Markdown using Ovis2.5-9B.",
    )
    parser.add_argument("input_file", help="Input Markdown file path")
    parser.add_argument("output_file", help="Output Markdown file path")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Enable dummy mode (fixed test data without API call)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=API_KEY,
        help="API key",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    content = input_path.read_text(encoding="utf-8")
    image_paths = find_markdown_images(content)

    if not image_paths:
        print("No local images found")
        return 0

    print(f"Found {len(image_paths)} local images")

    analyzer = ImageAnalyzer(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        dummy_mode=args.dummy,
    )

    results = analyzer.analyze_images([Path(p) for p in image_paths])

    if results:
        new_content = update_markdown_images(content, results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(new_content, encoding="utf-8")
        print(f"Saved to: {output_path}")
    else:
        print("No images were successfully processed")

    return 0
