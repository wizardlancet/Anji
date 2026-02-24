"""AST enhancement operations.

This module provides utilities for enhancing Markdown AST,
including heading level fixes, image filtering, and more.
"""

from __future__ import annotations

from typing import Any, Optional
from chinvat.ast_handler import MarkdownAST
from chinvat.image_analyzer import ImageAnalyzer


class Enhancer:
    """Enhancer for Markdown AST.

    This class provides methods to enhance Markdown AST by:
    - Fixing heading levels
    - Filtering decorative images
    - Enriching image nodes with analysis results

    Examples:
        >>> enhancer = Enhancer()
        >>> enhanced_ast = enhancer.enhance(ast, images_dir="path/to/images")
    """

    def __init__(
        self,
        image_analyzer: Optional[ImageAnalyzer] = None,
        ast_handler: Optional[MarkdownAST] = None,
    ):
        """Create an enhancer.

        Args:
            image_analyzer: Optional ImageAnalyzer for image analysis.
            ast_handler: Optional MarkdownAST handler.
        """
        self.image_analyzer = image_analyzer
        self.ast_handler = ast_handler or MarkdownAST()

    def fix_heading_levels(
        self,
        tokens: list[dict[str, Any]],
        min_level: int = 1,
        max_level: int = 6,
    ) -> list[dict[str, Any]]:
        """Fix heading levels in the AST.

        Ensures headings are sequential and properly nested.

        Args:
            tokens: The AST tokens.
            min_level: Minimum heading level (default: 1).
            max_level: Maximum heading level (default: 6).

        Returns:
            The enhanced AST tokens.
        """
        current_level = 0
        result = []

        for token in tokens:
            if token["type"] == "heading":
                level = token["attrs"]["level"]
                # Ensure heading is not too deep
                if level > max_level:
                    level = max_level
                # Ensure heading is not too shallow relative to context
                if level <= current_level:
                    level = min(current_level + 1, max_level)
                token["attrs"]["level"] = level
                current_level = level
            elif token["type"] == "list":
                # Reset level after lists
                current_level = min(current_level, 1)
            result.append(token)

        return result

    def filter_decorative_images(
        self,
        tokens: list[dict[str, Any]],
        images_dir: Optional[str] = None,
        precomputed_results: Optional[dict[str, dict]] = None,
    ) -> list[dict[str, Any]]:
        """Filter out decorative images from the AST.

        Args:
            tokens: The AST tokens.
            images_dir: Optional directory containing images for analysis.
            precomputed_results: Optional precomputed image analysis results to avoid duplicate API calls.

        Returns:
            The filtered AST tokens.
        """
        import asyncio

        images = self.ast_handler.find_images(tokens)

        if not images:
            return tokens

        # Get image paths that need filtering
        non_informative = set()
        if self.image_analyzer and images_dir:
            from pathlib import Path

            if precomputed_results is not None:
                # Use precomputed results
                results = precomputed_results
            else:
                # Analyze images if not already done
                image_paths = []
                for img in images:
                    url = img.get("attrs", {}).get("url", "")
                    if url:
                        image_paths.append(Path(images_dir) / url)

                results = asyncio.run(self.image_analyzer.analyze_images(image_paths))

            for path, result in results.items():
                # path is a string from the dict key
                path_obj = Path(path)
                if not result.get("is_informative", True):
                    non_informative.add(path_obj.stem if path_obj.suffix else str(path))

        # Filter tokens
        return self._filter_images_from_tokens(tokens, non_informative)

    def _filter_images_from_tokens(
        self,
        tokens: list[dict[str, Any]],
        non_informative: set[str],
    ) -> list[dict[str, Any]]:
        """Remove non-informative images from tokens.

        Args:
            tokens: The AST tokens.
            non_informative: Set of image identifiers to filter.

        Returns:
            Filtered tokens.
        """
        result = []
        for token in tokens:
            if token["type"] == "image":
                url = token.get("attrs", {}).get("url", "")
                # Check if this image should be filtered
                should_filter = False
                for identifier in non_informative:
                    if identifier in url:
                        should_filter = True
                        break
                if not should_filter:
                    result.append(token)
            elif "children" in token:
                token["children"] = self._filter_images_from_tokens(
                    token["children"], non_informative
                )
                result.append(token)
            else:
                result.append(token)
        return result

    def enrich_images_with_analysis(
        self,
        tokens: list[dict[str, Any]],
        images_dir: Optional[str] = None,
        update_content: bool = False,
        precomputed_results: Optional[dict[str, dict]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict]]:
        """Enrich image nodes with analysis results.

        Args:
            tokens: The AST tokens.
            images_dir: Optional directory containing images.
            update_content: Whether to update image children with captions.
            precomputed_results: Optional precomputed image analysis results to avoid duplicate API calls.

        Returns:
            A tuple of (enhanced AST tokens, analysis results dict).
        """
        import asyncio

        images = self.ast_handler.find_images(tokens)

        if not images or not self.image_analyzer or not images_dir:
            return tokens, {}

        from pathlib import Path

        if precomputed_results is not None:
            # Use precomputed results
            results = precomputed_results
        else:
            # Analyze images if not already done
            image_paths = []
            for img in images:
                url = img.get("attrs", {}).get("url", "")
                if url:
                    image_paths.append(Path(images_dir) / url)

            results = asyncio.run(self.image_analyzer.analyze_images(image_paths))

        # Update AST tokens with analysis results
        for token in images:
            url = token.get("attrs", {}).get("url", "")
            path = Path(images_dir) / url if url else None
            path_str = str(path) if path else ""

            if path_str in results:
                result = results[path_str]
                attrs = token.get("attrs", {})

                # Add analysis fields to token
                if "caption" in result:
                    attrs["caption"] = result["caption"]
                if "description" in result:
                    attrs["desc"] = result["description"]
                if "diagram" in result:
                    attrs["flowchart"] = result["diagram"]
                if "is_informative" in result:
                    attrs["is_informative"] = result["is_informative"]

                token["attrs"] = attrs

                # Optionally update the alt text
                if update_content and "caption" in result:
                    token["children"] = [{"type": "text", "raw": result["caption"]}]

        return tokens, results

    def enhance(
        self,
        tokens: list[dict[str, Any]],
        images_dir: Optional[str] = None,
        fix_headings: bool = True,
        filter_decorative: bool = True,
        enrich_images: bool = True,
    ) -> list[dict[str, Any]]:
        """Apply all enhancement operations to the AST.

        Args:
            tokens: The AST tokens.
            images_dir: Optional directory containing images.
            fix_headings: Whether to fix heading levels.
            filter_decorative: Whether to filter decorative images.
            enrich_images: Whether to enrich images with analysis.

        Returns:
            The enhanced AST tokens.
        """
        import asyncio
        from pathlib import Path

        # Determine if we need image analysis
        needs_analysis = (
            self.image_analyzer
            and images_dir
            and (enrich_images or filter_decorative)
        )

        # Precompute image analysis results once to avoid duplicate API calls
        precomputed_results: dict[str, dict] = {}
        if needs_analysis:
            images = self.ast_handler.find_images(tokens)
            if images and images_dir:
                image_paths = []
                for img in images:
                    url = img.get("attrs", {}).get("url", "")
                    if url:
                        image_paths.append(Path(images_dir) / url)

                if image_paths and self.image_analyzer:
                    precomputed_results = asyncio.run(
                        self.image_analyzer.analyze_images(image_paths)
                    )

        # Step 1: Enrich images with analysis (needs to happen before filtering)
        if enrich_images and precomputed_results:
            tokens, _ = self.enrich_images_with_analysis(
                tokens, images_dir, update_content=False, precomputed_results=precomputed_results
            )
        elif enrich_images and self.image_analyzer:
            # Fallback: analyze if not done yet (for backward compatibility)
            tokens, _ = self.enrich_images_with_analysis(
                tokens, images_dir, update_content=False
            )

        # Step 2: Filter decorative images
        if filter_decorative and precomputed_results:
            tokens = self.filter_decorative_images(
                tokens, images_dir, precomputed_results=precomputed_results
            )
        elif filter_decorative and self.image_analyzer:
            # Fallback: analyze if not done yet (for backward compatibility)
            tokens = self.filter_decorative_images(tokens, images_dir)

        # Step 3: Fix heading levels
        if fix_headings:
            tokens = self.fix_heading_levels(tokens)

        return tokens


def enhance_markdown(
    markdown: str,
    images_dir: Optional[str] = None,
    image_analyzer: Optional[ImageAnalyzer] = None,
    fix_headings: bool = True,
    filter_decorative: bool = True,
    enrich_images: bool = True,
) -> str:
    """Convenience function to enhance Markdown content.

    Args:
        markdown: The Markdown content to enhance.
        images_dir: Optional directory containing images.
        image_analyzer: Optional ImageAnalyzer instance.
        fix_headings: Whether to fix heading levels.
        filter_decorative: Whether to filter decorative images.
        enrich_images: Whether to enrich images with analysis.

    Returns:
        The enhanced Markdown content.
    """
    ast_handler = MarkdownAST()
    enhancer = Enhancer(image_analyzer=image_analyzer, ast_handler=ast_handler)

    tokens = ast_handler.parse(markdown)
    enhanced_tokens = enhancer.enhance(
        tokens,
        images_dir=images_dir,
        fix_headings=fix_headings,
        filter_decorative=filter_decorative,
        enrich_images=enrich_images,
    )

    return ast_handler.render(enhanced_tokens)
