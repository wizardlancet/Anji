"""Full pipeline orchestration.

This module provides the main pipeline for converting PDFs to enhanced
Markdown/JSON documents optimized for AI agent consumption.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

from chinvat.pdf_converter import (
    PDFToMarkdownConverter,
    get_default_converter,
)
from chinvat.ast_handler import MarkdownAST
from chinvat.image_analyzer import ImageAnalyzer, get_default_analyzer
from chinvat.enhancement import Enhancer
from chinvat.exporters import (
    export_to_markdown,
    export_to_json,
    export_to_structured_data,
)


class Pipeline:
    """The main Chinvat pipeline for PDF to enhanced document conversion.

    This pipeline orchestrates the full conversion process:
    1. PDF -> raw Markdown (using PaddleOCR-VL)
    2. Markdown -> AST (using mistune)
    3. AST -> enhanced AST (image analysis, heading fixes, filtering)
    4. AST -> output format (Markdown, JSON, or structured data)

    Examples:
        >>> pipeline = Pipeline()
        >>> result = pipeline.run("input.pdf", "output_dir")

        >>> # With custom settings
        >>> pipeline = Pipeline(
        ...     paddleocr_server_url="http://localhost:8118/v1",
        ...     vlm_server_url="http://localhost:8000/v1",
        ... )
        >>> pipeline.run("input.pdf", "output_dir", output_format="json")
    """

    def __init__(
        self,
        paddleocr_server_url: str = "http://localhost:8118/v1",
        paddleocr_backend: str = "vllm-server",
        vlm_server_url: str = "http://localhost:8000/v1",
        vlm_api_key: str = "abc-123",
        vlm_model_name: str = "AIDC-AI/Ovis2.5-9B",
    ):
        """Create a pipeline instance.

        Args:
            paddleocr_server_url: PaddleOCR-VL server URL.
            paddleocr_backend: PaddleOCR backend type.
            vlm_server_url: VLM server URL for image analysis.
            vlm_api_key: VLM API key.
            vlm_model_name: VLM model name.
        """
        self.paddleocr_server_url = paddleocr_server_url
        self.paddleocr_backend = paddleocr_backend
        self.vlm_server_url = vlm_server_url
        self.vlm_api_key = vlm_api_key
        self.vlm_model_name = vlm_model_name

        self._pdf_converter: Optional[PDFToMarkdownConverter] = None
        self._image_analyzer: Optional[ImageAnalyzer] = None

    @property
    def pdf_converter(self) -> PDFToMarkdownConverter:
        """Get or create the PDF converter."""
        if self._pdf_converter is None:
            self._pdf_converter = PDFToMarkdownConverter(
                server_url=self.paddleocr_server_url,
                backend=self.paddleocr_backend,
            )
        return self._pdf_converter

    @property
    def image_analyzer(self) -> ImageAnalyzer:
        """Get or create the image analyzer."""
        if self._image_analyzer is None:
            self._image_analyzer = ImageAnalyzer(
                api_base_url=self.vlm_server_url,
                api_key=self.vlm_api_key,
                model_name=self.vlm_model_name,
            )
        return self._image_analyzer

    def run(
        self,
        input_path: Union[str, Path],
        output_folder: Union[str, Path],
        output_format: str = "markdown",
        enhance: bool = True,
        fix_headings: bool = True,
        filter_decorative: bool = True,
        enrich_images: bool = True,
        merge_tables: bool = True,
        relevel_titles: bool = True,
        concatenate_pages: bool = True,
        cleanup: bool = True,
        keep_images: bool = True,
        embed_base64: bool = False,
    ) -> dict[str, str]:
        """Run the full pipeline on a single PDF.

        Args:
            input_path: Path to the input PDF.
            output_folder: Path to the output folder.
            output_format: Output format ('markdown', 'json', 'structured').
            enhance: Whether to apply enhancements.
            fix_headings: Whether to fix heading levels.
            filter_decorative: Whether to filter decorative images.
            enrich_images: Whether to enrich images with analysis.
            merge_tables: Whether to merge cross-page tables.
            relevel_titles: Whether to rebuild headings in OCR.
            concatenate_pages: Whether to concatenate pages.
            cleanup: Whether to remove intermediate files.
            keep_images: Whether to keep images folder in output (default True).
            embed_base64: Whether to embed images as base64 in markdown (default False).

        Returns:
            A dictionary with paths to output files.
        """
        input_path = Path(input_path)
        output_folder = Path(output_folder)
        base_folder = output_folder / input_path.stem

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Step 1: Convert PDF to raw Markdown
        print(f"[1/4] Converting PDF to Markdown: {input_path}")
        raw_folder = base_folder / "raw"
        self.pdf_converter.convert(
            input_path=input_path,
            output_folder=raw_folder,
            merge_tables=merge_tables,
            relevel_titles=relevel_titles,
            concatenate_pages=concatenate_pages,
        )

        # Find the raw markdown file
        md_files = list(raw_folder.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No Markdown file found in {raw_folder}")
        raw_markdown_path = md_files[0]

        # Read raw Markdown
        raw_content = raw_markdown_path.read_text(encoding="utf-8")

        # Step 2: Parse Markdown to AST
        print(f"[2/4] Parsing Markdown to AST")
        ast_handler = MarkdownAST()
        tokens = ast_handler.parse(raw_content)

        # Step 3: Enhance AST (if requested)
        if enhance:
            print(f"[3/4] Enhancing AST")
            enhancer = Enhancer(
                image_analyzer=self.image_analyzer,
                ast_handler=ast_handler,
            )
            images_dir = str(raw_folder)
            tokens = enhancer.enhance(
                tokens,
                images_dir=images_dir,
                fix_headings=fix_headings,
                filter_decorative=filter_decorative,
                enrich_images=enrich_images,
            )

        # Step 4: Export to output format
        print(f"[4/4] Exporting to {output_format}")
        enhanced_folder = base_folder / "enhanced"
        enhanced_folder.mkdir(parents=True, exist_ok=True)

        # Handle images folder
        raw_images_dir = raw_folder / "imgs"
        enhanced_images_dir = enhanced_folder / "imgs"

        if keep_images and raw_images_dir.exists():
            if enhanced_images_dir.exists():
                shutil.rmtree(enhanced_images_dir)
            shutil.copytree(raw_images_dir, enhanced_images_dir)

        outputs = {}

        if output_format in ("markdown", "both"):
            md_output = enhanced_folder / "document.md"
            # Always pass raw_folder for base64 embedding (images exist there even if not copied to enhanced)
            # When keep_images is True: copy raw imgs to enhanced imgs, use enhanced imgs for base64
            # When keep_images is False: use raw imgs for base64 (they'll be cleaned up later if cleanup=True)
            images_dir = str(raw_images_dir) if raw_images_dir.exists() else None
            export_to_markdown(
                tokens,
                str(md_output),
                embed_base64=embed_base64,
                images_dir=images_dir,
            )
            outputs["markdown"] = str(md_output)

        if output_format in ("json", "both"):
            json_output = enhanced_folder / "document.json"
            export_to_json(tokens, str(json_output))
            outputs["json"] = str(json_output)

        if output_format == "structured":
            structured_output = enhanced_folder / "document_structured.json"
            export_to_structured_data(tokens, str(structured_output))
            outputs["structured"] = str(structured_output)

        # Cleanup intermediate files if requested
        if cleanup:
            if raw_folder.exists():
                shutil.rmtree(raw_folder)

        return outputs

    def run_batch(
        self,
        input_paths: list[Union[str, Path]],
        output_base_folder: Union[str, Path],
        auto_naming: bool = True,
        **kwargs,
    ) -> list[dict[str, str]]:
        """Run the pipeline on multiple PDFs.

        Args:
            input_paths: List of input PDF paths.
            output_base_folder: Base output folder.
            auto_naming: Whether to name output folders by PDF stem.
            **kwargs: Additional arguments passed to run().

        Returns:
            A list of output dictionaries for each PDF.
        """
        output_base = Path(output_base_folder)
        results = []

        for idx, input_path in enumerate(input_paths, start=1):
            input_path = Path(input_path)

            if auto_naming:
                output_folder = output_base / input_path.stem
            else:
                output_folder = output_base / str(idx)

            print(f"\n{'='*50}")
            print(f"Processing: {input_path.name}")
            print(f"{'='*50}")

            result = self.run(input_path, output_folder, **kwargs)
            results.append(result)

        return results

    def close(self) -> None:
        """Close the pipeline and release resources."""
        if self._pdf_converter is not None:
            self._pdf_converter.close()
            self._pdf_converter = None
        if self._image_analyzer is not None:
            self._image_analyzer.close()
            self._image_analyzer = None


def run_full_pipeline(
    input_path: Union[str, Path],
    output_folder: Union[str, Path],
    paddleocr_server_url: str = "http://localhost:8118/v1",
    vlm_server_url: str = "http://localhost:8000/v1",
    **kwargs,
) -> dict[str, str]:
    """Convenience function to run the full pipeline.

    Args:
        input_path: Path to the input PDF.
        output_folder: Path to the output folder.
        paddleocr_server_url: PaddleOCR-VL server URL.
        vlm_server_url: VLM server URL for image analysis.
        **kwargs: Additional arguments passed to Pipeline.run().

    Returns:
        A dictionary with paths to output files.
    """
    pipeline = Pipeline(
        paddleocr_server_url=paddleocr_server_url,
        vlm_server_url=vlm_server_url,
    )
    try:
        return pipeline.run(input_path, output_folder, **kwargs)
    finally:
        pipeline.close()


def batch_pipeline(
    input_paths: list[Union[str, Path]],
    output_base_folder: Union[str, Path],
    paddleocr_server_url: str = "http://localhost:8118/v1",
    vlm_server_url: str = "http://localhost:8000/v1",
    auto_naming: bool = True,
    **kwargs,
) -> list[dict[str, str]]:
    """Convenience function to run the pipeline on multiple PDFs.

    Args:
        input_paths: List of input PDF paths.
        output_base_folder: Base output folder.
        paddleocr_server_url: PaddleOCR-VL server URL.
        vlm_server_url: VLM server URL.
        auto_naming: Whether to name output folders by PDF stem.
        **kwargs: Additional arguments passed to Pipeline.run().

    Returns:
        A list of output dictionaries for each PDF.
    """
    pipeline = Pipeline(
        paddleocr_server_url=paddleocr_server_url,
        vlm_server_url=vlm_server_url,
    )
    try:
        return pipeline.run_batch(
            input_paths=input_paths,
            output_base_folder=output_base_folder,
            auto_naming=auto_naming,
            **kwargs,
        )
    finally:
        pipeline.close()
