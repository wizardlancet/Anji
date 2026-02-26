"""PDF to Markdown conversion using PaddleOCR-VL.

This module provides utilities for converting PDF files to Markdown format
using the PaddleOCR-VL visual language model server.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from paddleocr import PaddleOCRVL


class PDFToMarkdownConverter:
    """A PDF-to-Markdown converter using PaddleOCR-VL.

    This class provides an interface to convert PDF files to Markdown format
    using the PaddleOCR-VL visual language model. It supports lazy initialization
    and singleton pattern for efficient batch processing.

    Examples:
        >>> converter = PDFToMarkdownConverter()
        >>> converter.convert("input.pdf", "./output")
        >>> # Or use the convenience function
        >>> from anji import pdf_to_markdown
        >>> pdf_to_markdown("input.pdf", "./output")
    """

    _instance: Optional["PDFToMarkdownConverter"] = None
    _pipeline: Optional[PaddleOCRVL] = None

    def __init__(
        self,
        server_url: str = "http://localhost:8118/v1",
        backend: str = "vllm-server",
    ):
        """Create a converter.

        Args:
            server_url: VLM server URL. Defaults to "http://localhost:8118/v1".
            backend: VL recognition backend. Defaults to "vllm-server".
        """
        self.server_url = server_url
        self.backend = backend
        self._pipeline = None

    @property
    def pipeline(self) -> "PaddleOCRVL":
        """Lazily initialize the PaddleOCR pipeline."""
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(
                vl_rec_backend=self.backend,
                vl_rec_server_url=self.server_url
            )
        return self._pipeline

    def convert(
        self,
        input_path: Union[str, Path],
        output_folder: Union[str, Path],
        merge_tables: bool = True,
        relevel_titles: bool = True,
        concatenate_pages: bool = True,
        print_result: bool = False,
    ) -> str:
        """Convert a PDF file to Markdown.

        Args:
            input_path: Input PDF file path.
            output_folder: Output folder path.
            merge_tables: Whether to merge cross-page tables.
            relevel_titles: Whether to rebuild multi-level headings.
            concatenate_pages: Whether to concatenate multi-page results.
            print_result: Whether to print structured results.

        Returns:
            The output folder path.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input file is not a PDF.
        """
        input_path = Path(input_path)
        output_folder = Path(output_folder)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file must be a PDF: {input_path}")

        output_folder.mkdir(parents=True, exist_ok=True)

        output = self.pipeline.predict(input=str(input_path))
        pages_res = list(output)

        output = self.pipeline.restructure_pages(
            pages_res,
            merge_tables=merge_tables,
            relevel_titles=relevel_titles,
            concatenate_pages=concatenate_pages
        )

        for res in output:
            if print_result:
                res.print()
            res.save_to_markdown(save_path=str(output_folder), pretty=False)

        return str(output_folder)

    def close(self) -> None:
        """Close the converter and release resources."""
        if self._pipeline is not None:
            self._pipeline = None


# Global default converter instance (lazy init)
_default_converter: Optional[PDFToMarkdownConverter] = None


def get_default_converter(
    server_url: str = "http://localhost:8118/v1",
    backend: str = "vllm-server",
) -> PDFToMarkdownConverter:
    """Get the default converter (singleton).

    Args:
        server_url: VLM server URL.
        backend: VL recognition backend.

    Returns:
        A PDFToMarkdownConverter instance.
    """
    global _default_converter
    if (
        _default_converter is None
        or _default_converter.server_url != server_url
        or _default_converter.backend != backend
    ):
        _default_converter = PDFToMarkdownConverter(
            server_url=server_url,
            backend=backend
        )
    return _default_converter


def pdf_to_markdown(
    input_path: Union[str, Path],
    output_folder: Union[str, Path],
    merge_tables: bool = True,
    relevel_titles: bool = True,
    concatenate_pages: bool = True,
    server_url: str = "http://localhost:8118/v1",
    backend: str = "vllm-server",
    print_result: bool = False,
) -> str:
    """Convenience wrapper to convert a PDF file to Markdown.

    Args:
        input_path: Input PDF file path.
        output_folder: Output folder path.
        merge_tables: Whether to merge cross-page tables. Defaults to True.
        relevel_titles: Whether to rebuild multi-level headings. Defaults to True.
        concatenate_pages: Whether to concatenate multi-page results. Defaults to True.
        server_url: VLM server URL. Defaults to "http://localhost:8118/v1".
        backend: VL recognition backend. Defaults to "vllm-server".
        print_result: Whether to print structured results. Defaults to False.

    Returns:
        The output folder path.

    Example:
        >>> pdf_to_markdown("document.pdf", "./output")
        './output'
    """
    converter = get_default_converter(server_url=server_url, backend=backend)
    return converter.convert(
        input_path=input_path,
        output_folder=output_folder,
        merge_tables=merge_tables,
        relevel_titles=relevel_titles,
        concatenate_pages=concatenate_pages,
        print_result=print_result,
    )


def batch_pdf_to_markdown(
    input_paths: list[Union[str, Path]],
    output_base_folder: Union[str, Path],
    auto_naming: bool = True,
    **kwargs: Any,
) -> list[str]:
    """Batch convert multiple PDF files to Markdown.

    Args:
        input_paths: List of input PDF file paths.
        output_base_folder: Base output folder path.
        auto_naming: If True, create subfolders named after each PDF stem.
        **kwargs: Extra keyword args forwarded to pdf_to_markdown.

    Returns:
        A list of output folder paths.

    Example:
        >>> pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        >>> batch_pdf_to_markdown(pdfs, "./outputs")
        ['./outputs/doc1', './outputs/doc2', './outputs/doc3']
    """
    output_base = Path(output_base_folder)
    results: list[str] = []
    converter = get_default_converter(**kwargs)

    for input_path in input_paths:
        input_path = Path(input_path)

        if auto_naming:
            output_folder = output_base / input_path.stem
        else:
            output_folder = output_base / str(len(results) + 1)

        result = converter.convert(
            input_path=input_path,
            output_folder=output_folder,
            merge_tables=kwargs.get("merge_tables", True),
            relevel_titles=kwargs.get("relevel_titles", True),
            concatenate_pages=kwargs.get("concatenate_pages", True),
            print_result=kwargs.get("print_result", False),
        )
        results.append(result)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using PaddleOCR VL.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--server-url",
            type=str,
            default="http://localhost:8118/v1",
            help='VLM server URL (default: "http://localhost:8118/v1").',
        )
        p.add_argument(
            "--backend",
            type=str,
            default="vllm-server",
            help='VL recognition backend (default: "vllm-server").',
        )
        p.add_argument(
            "--no-merge-tables",
            action="store_true",
            help="Disable merging cross-page tables.",
        )
        p.add_argument(
            "--no-relevel-titles",
            action="store_true",
            help="Disable rebuilding multi-level headings.",
        )
        p.add_argument(
            "--no-concatenate-pages",
            action="store_true",
            help="Disable concatenating multi-page results into one.",
        )
        p.add_argument(
            "--print-result",
            action="store_true",
            help="Print structured results to stdout.",
        )

    single = subparsers.add_parser(
        "single",
        help="Convert one PDF file.",
    )
    single.add_argument(
        "input_file",
        type=str,
        help="Path to the input PDF file.",
    )
    single.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder where Markdown will be written.",
    )
    add_common_options(single)

    batch = subparsers.add_parser(
        "batch",
        help="Convert multiple PDF files.",
    )
    batch.add_argument(
        "output_base_folder",
        type=str,
        help="Base output folder (subfolders will be created under it).",
    )
    batch.add_argument(
        "input_files",
        nargs="*",
        type=str,
        help="One or more input PDF files.",
    )
    batch.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Optional directory to scan for PDFs.",
    )
    batch.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help='Glob pattern used with --input-dir (default: "*.pdf").',
    )
    batch.add_argument(
        "--recursive",
        action="store_true",
        help="Scan --input-dir recursively.",
    )
    batch.add_argument(
        "--no-auto-naming",
        action="store_true",
        help="Disable naming output folders by PDF stem (use 1,2,3,... instead).",
    )
    add_common_options(batch)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for PDF to Markdown conversion."""
    parser = _build_arg_parser()
    argv_list = list(argv) if argv is not None else None

    if argv_list is None:
        import sys
        argv_list = sys.argv[1:]

    if argv_list and argv_list[0] not in {"single", "batch", "-h", "--help"}:
        argv_list = ["single", *argv_list]

    args = parser.parse_args(argv_list)

    merge_tables = not args.no_merge_tables
    relevel_titles = not args.no_relevel_titles
    concatenate_pages = not args.no_concatenate_pages

    if args.command == "single":
        converter = PDFToMarkdownConverter(server_url=args.server_url, backend=args.backend)
        converter.convert(
            input_path=args.input_file,
            output_folder=args.output_folder,
            merge_tables=merge_tables,
            relevel_titles=relevel_titles,
            concatenate_pages=concatenate_pages,
            print_result=args.print_result,
        )
        return 0

    if args.command == "batch":
        input_files: list[str] = list(args.input_files or [])

        if args.input_dir:
            input_dir = Path(args.input_dir)
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
            if not input_dir.is_dir():
                raise NotADirectoryError(f"Input directory is not a directory: {input_dir}")

            candidates = (
                input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern)
            )
            input_files.extend(str(p) for p in candidates if p.is_file())

        seen: set[str] = set()
        unique_input_files: list[str] = []
        for p in input_files:
            if p not in seen:
                unique_input_files.append(p)
                seen.add(p)

        if not unique_input_files:
            raise ValueError(
                "No input PDFs provided. Pass files as positional args, or use --input-dir."
            )

        converter = PDFToMarkdownConverter(server_url=args.server_url, backend=args.backend)
        output_base = Path(args.output_base_folder)
        output_base.mkdir(parents=True, exist_ok=True)

        results: list[str] = []
        for idx, input_path_str in enumerate(unique_input_files, start=1):
            input_path = Path(input_path_str)
            if args.no_auto_naming:
                output_folder = output_base / str(idx)
            else:
                output_folder = output_base / input_path.stem

            results.append(
                converter.convert(
                    input_path=input_path,
                    output_folder=output_folder,
                    merge_tables=merge_tables,
                    relevel_titles=relevel_titles,
                    concatenate_pages=concatenate_pages,
                    print_result=args.print_result,
                )
            )

        return 0

    raise AssertionError(f"Unhandled command: {args.command}")
