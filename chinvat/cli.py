"""Command-line interface for chinvat.

This module provides CLI commands for all pipeline stages and the
full end-to-end conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from chinvat.pipeline import Pipeline
from chinvat.pdf_converter import PDFToMarkdownConverter, main as pdf_main
from chinvat.ast_handler import MarkdownAST
from chinvat.image_analyzer import ImageAnalyzer, main as img_main
from chinvat.enhancement import enhance_markdown
from chinvat.exporters import (
    export_to_markdown,
    export_to_json,
    export_to_structured_data,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    parser.add_argument(
        "--paddleocr-url",
        type=str,
        default="http://localhost:8118/v1",
        help="PaddleOCR-VL server URL",
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLM server URL for image analysis",
    )


def build_pipeline_parser() -> argparse.ArgumentParser:
    """Build the pipeline subcommand parser."""
    parser = argparse.ArgumentParser(
        description="Run the full chinvat pipeline",
    )
    add_common_args(parser)

    parser.add_argument(
        "input",
        type=str,
        help="Input PDF file path",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output folder path",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "structured", "both"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable enhancement step",
    )
    parser.add_argument(
        "--no-fix-headings",
        action="store_true",
        help="Don't fix heading levels",
    )
    parser.add_argument(
        "--no-filter-decorative",
        action="store_true",
        help="Don't filter decorative images",
    )
    parser.add_argument(
        "--no-enrich-images",
        action="store_true",
        help="Don't enrich images with analysis",
    )
    parser.add_argument(
        "--no-merge-tables",
        action="store_true",
        help="Don't merge cross-page tables",
    )
    parser.add_argument(
        "--no-relevel-titles",
        action="store_true",
        help="Don't rebuild headings in OCR",
    )
    parser.add_argument(
        "--no-concatenate",
        action="store_true",
        help="Don't concatenate pages",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't remove intermediate files",
    )
    parser.add_argument(
        "--no-keep-images",
        action="store_true",
        help="Don't keep images folder in output (default: keep)",
    )
    parser.add_argument(
        "--embed-base64",
        action="store_true",
        help="Embed images as base64 data URLs in markdown",
    )

    return parser


def build_batch_parser() -> argparse.ArgumentParser:
    """Build the batch processing subcommand parser."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple PDFs",
    )
    add_common_args(parser)

    parser.add_argument(
        "output_base",
        type=str,
        help="Base output folder",
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        type=str,
        help="Input PDF files",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to scan for PDFs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern for PDF files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directories recursively",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "structured", "both"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--no-auto-naming",
        action="store_true",
        help="Use sequential naming (1, 2, ...) instead of filenames",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't remove intermediate files",
    )
    parser.add_argument(
        "--no-keep-images",
        action="store_true",
        help="Don't keep images folder in output (default: keep)",
    )
    parser.add_argument(
        "--embed-base64",
        action="store_true",
        help="Embed images as base64 data URLs in markdown",
    )

    return parser


def build_markdown_parser() -> argparse.ArgumentParser:
    """Build the markdown processing subcommand parser."""
    parser = argparse.ArgumentParser(
        description="Process Markdown files (AST parsing, enhancement, export)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parse subcommand
    parse_cmd = subparsers.add_parser("parse", help="Parse Markdown to AST")
    parse_cmd.add_argument(
        "input",
        type=str,
        help="Input Markdown file",
    )
    parse_cmd.add_argument(
        "--output",
        type=str,
        help="Output JSON file for AST",
    )

    # Enhance subcommand
    enhance_cmd = subparsers.add_parser(
        "enhance",
        help="Enhance Markdown with image analysis",
    )
    enhance_cmd.add_argument(
        "input",
        type=str,
        help="Input Markdown file",
    )
    enhance_cmd.add_argument(
        "output",
        type=str,
        help="Output Markdown file",
    )
    enhance_cmd.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images",
    )
    enhance_cmd.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLM server URL",
    )
    enhance_cmd.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy mode",
    )

    # Export subcommand
    export_cmd = subparsers.add_parser(
        "export",
        help="Export AST to different formats",
    )
    export_cmd.add_argument(
        "input",
        type=str,
        help="Input Markdown file or AST JSON",
    )
    export_cmd.add_argument(
        "output",
        type=str,
        help="Output file",
    )
    export_cmd.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "structured"],
        default="markdown",
        help="Output format",
    )

    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the main CLI parser."""
    parser = argparse.ArgumentParser(
        prog="chinvat",
        description="Chinvat - PDF to AI Agent Knowledge Bridge",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full pipeline",
    )
    add_common_args(pipeline_parser)
    pipeline_parser.add_argument(
        "input",
        type=str,
        help="Input PDF file path",
    )
    pipeline_parser.add_argument(
        "output",
        type=str,
        help="Output folder path",
    )
    pipeline_parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "structured", "both"],
        default="markdown",
        help="Output format",
    )
    pipeline_parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable enhancement step",
    )
    pipeline_parser.add_argument(
        "--no-fix-headings",
        action="store_true",
        help="Don't fix heading levels",
    )
    pipeline_parser.add_argument(
        "--no-filter-decorative",
        action="store_true",
        help="Don't filter decorative images",
    )
    pipeline_parser.add_argument(
        "--no-enrich-images",
        action="store_true",
        help="Don't enrich images with analysis",
    )
    pipeline_parser.add_argument(
        "--no-merge-tables",
        action="store_true",
        help="Don't merge cross-page tables",
    )
    pipeline_parser.add_argument(
        "--no-relevel-titles",
        action="store_true",
        help="Don't rebuild headings in OCR",
    )
    pipeline_parser.add_argument(
        "--no-concatenate",
        action="store_true",
        help="Don't concatenate pages",
    )
    pipeline_parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't remove intermediate files",
    )
    pipeline_parser.add_argument(
        "--no-keep-images",
        action="store_true",
        help="Don't keep images folder in output (default: keep)",
    )
    pipeline_parser.add_argument(
        "--embed-base64",
        action="store_true",
        help="Embed images as base64 data URLs in markdown",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process multiple PDFs",
    )
    add_common_args(batch_parser)
    batch_parser.add_argument(
        "output_base",
        type=str,
        help="Base output folder",
    )
    batch_parser.add_argument(
        "input_files",
        nargs="*",
        type=str,
        help="Input PDF files",
    )
    batch_parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to scan for PDFs",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern for PDF files",
    )
    batch_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directories recursively",
    )
    batch_parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "structured", "both"],
        default="markdown",
        help="Output format",
    )
    batch_parser.add_argument(
        "--no-auto-naming",
        action="store_true",
        help="Use sequential naming",
    )
    batch_parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable enhancement step",
    )
    batch_parser.add_argument(
        "--no-fix-headings",
        action="store_true",
        help="Don't fix heading levels",
    )
    batch_parser.add_argument(
        "--no-filter-decorative",
        action="store_true",
        help="Don't filter decorative images",
    )
    batch_parser.add_argument(
        "--no-enrich-images",
        action="store_true",
        help="Don't enrich images with analysis",
    )
    batch_parser.add_argument(
        "--no-merge-tables",
        action="store_true",
        help="Don't merge cross-page tables",
    )
    batch_parser.add_argument(
        "--no-relevel-titles",
        action="store_true",
        help="Don't rebuild headings in OCR",
    )
    batch_parser.add_argument(
        "--no-concatenate",
        action="store_true",
        help="Don't concatenate pages",
    )
    batch_parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't remove intermediate files",
    )
    batch_parser.add_argument(
        "--no-keep-images",
        action="store_true",
        help="Don't keep images folder in output (default: keep)",
    )
    batch_parser.add_argument(
        "--embed-base64",
        action="store_true",
        help="Embed images as base64 data URLs in markdown",
    )

    # PDF command (wrapper around pdf_converter)
    pdf_parser = subparsers.add_parser(
        "pdf",
        help="Convert PDF to Markdown (step 1 only)",
    )
    pdf_parser.add_argument(
        "input",
        type=str,
        help="Input PDF file",
    )
    pdf_parser.add_argument(
        "output",
        type=str,
        help="Output folder",
    )
    pdf_parser.add_argument(
        "--paddleocr-url",
        type=str,
        default="http://localhost:8118/v1",
        help="PaddleOCR-VL server URL",
    )
    pdf_parser.add_argument(
        "--no-merge-tables",
        action="store_true",
        help="Don't merge tables",
    )
    pdf_parser.add_argument(
        "--no-relevel-titles",
        action="store_true",
        help="Don't rebuild titles",
    )

    # Markdown command
    md_parser = subparsers.add_parser(
        "md",
        help="Process Markdown files",
    )
    md_subparsers = md_parser.add_subparsers(dest="md_command", required=True)

    # MD Parse
    md_parse = md_subparsers.add_parser(
        "parse",
        help="Parse Markdown to AST",
    )
    md_parse.add_argument(
        "input",
        type=str,
        help="Input Markdown file",
    )
    md_parse.add_argument(
        "--output",
        type=str,
        help="Output JSON file",
    )

    # MD Enhance
    md_enhance = md_subparsers.add_parser(
        "enhance",
        help="Enhance Markdown",
    )
    md_enhance.add_argument(
        "input",
        type=str,
        help="Input Markdown file",
    )
    md_enhance.add_argument(
        "output",
        type=str,
        help="Output Markdown file",
    )
    md_enhance.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images",
    )
    md_enhance.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLM server URL",
    )
    md_enhance.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy mode",
    )

    # MD Export
    md_export = md_subparsers.add_parser(
        "export",
        help="Export to format",
    )
    md_export.add_argument(
        "input",
        type=str,
        help="Input Markdown file",
    )
    md_export.add_argument(
        "output",
        type=str,
        help="Output file",
    )
    md_export.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "structured"],
        default="markdown",
        help="Output format",
    )

    # Image command
    img_parser = subparsers.add_parser(
        "image",
        help="Analyze images (step 3 only)",
    )
    img_parser.add_argument(
        "input",
        type=str,
        help="Input Markdown file",
    )
    img_parser.add_argument(
        "output",
        type=str,
        help="Output Markdown file",
    )
    img_parser.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLM server URL",
    )
    img_parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy mode",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "pipeline":
        return _run_pipeline(args)

    elif args.command == "batch":
        return _run_batch(args)

    elif args.command == "pdf":
        return _run_pdf(args)

    elif args.command == "md":
        return _run_md(args)

    elif args.command == "image":
        return _run_image(args)

    else:
        parser.print_help()
        return 1


def _run_pipeline(args) -> int:
    """Run the full pipeline."""
    pipeline = Pipeline(
        paddleocr_server_url=args.paddleocr_url,
        vlm_server_url=args.vlm_url,
    )

    try:
        results = pipeline.run(
            input_path=args.input,
            output_folder=args.output,
            format=args.format,
            enhance=not args.no_enhance if hasattr(args, "no_enhance") else True,
            fix_headings=not args.no_fix_headings if hasattr(args, "no_fix_headings") else True,
            filter_decorative=not args.no_filter_decorative if hasattr(args, "no_filter_decorative") else True,
            enrich_images=not args.no_enrich_images if hasattr(args, "no_enrich_images") else True,
            merge_tables=not args.no_merge_tables if hasattr(args, "no_merge_tables") else True,
            relevel_titles=not args.no_relevel_titles if hasattr(args, "no_relevel_titles") else True,
            concatenate_pages=not args.no_concatenate if hasattr(args, "no_concatenate") else True,
            cleanup=not args.no_cleanup if hasattr(args, "no_cleanup") else True,
            keep_images=not args.no_keep_images if hasattr(args, "no_keep_images") else True,
            embed_base64=args.embed_base64 if hasattr(args, "embed_base64") else False,
        )

        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print("=" * 50)
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")

    finally:
        pipeline.close()

    return 0


def _run_batch(args) -> int:
    """Run batch processing."""
    input_files = list(args.input_files or [])

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return 1

        candidates = (
            input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern)
        )
        input_files.extend(str(p) for p in candidates if p.is_file())

    if not input_files:
        print("Error: No input files provided")
        return 1

    pipeline = Pipeline(
        paddleocr_server_url=args.paddleocr_url,
        vlm_server_url=args.vlm_url,
    )

    try:
        results = pipeline.run_batch(
            input_paths=input_files,
            output_base_folder=args.output_base,
            format=args.format,
            auto_naming=not args.no_auto_naming,
            enhance=not args.no_enhance if hasattr(args, "no_enhance") else True,
            fix_headings=not args.no_fix_headings if hasattr(args, "no_fix_headings") else True,
            filter_decorative=not args.no_filter_decorative if hasattr(args, "no_filter_decorative") else True,
            enrich_images=not args.no_enrich_images if hasattr(args, "no_enrich_images") else True,
            merge_tables=not args.no_merge_tables if hasattr(args, "no_merge_tables") else True,
            relevel_titles=not args.no_relevel_titles if hasattr(args, "no_relevel_titles") else True,
            concatenate_pages=not args.no_concatenate if hasattr(args, "no_concatenate") else True,
            cleanup=not args.no_cleanup if hasattr(args, "no_cleanup") else True,
            keep_images=not args.no_keep_images if hasattr(args, "no_keep_images") else True,
            embed_base64=args.embed_base64 if hasattr(args, "embed_base64") else False,
        )

        print("\n" + "=" * 50)
        print(f"Batch processing completed: {len(results)} files")
        print("=" * 50)

    finally:
        pipeline.close()

    return 0


def _run_pdf(args) -> int:
    """Run PDF conversion only."""
    converter = PDFToMarkdownConverter(server_url=args.paddleocr_url)
    converter.convert(
        input_path=args.input,
        output_folder=args.output,
        merge_tables=not args.no_merge_tables,
        relevel_titles=not args.no_relevel_titles,
    )
    return 0


def _run_md(args) -> int:
    """Run Markdown processing."""
    if args.md_command == "parse":
        ast_handler = MarkdownAST()
        content = Path(args.input).read_text(encoding="utf-8")
        tokens = ast_handler.parse(content)

        if args.output:
            import json
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(tokens, f, ensure_ascii=False, indent=2)
            print(f"Saved AST to: {args.output}")
        else:
            import json
            print(json.dumps(tokens, ensure_ascii=False, indent=2))

    elif args.md_command == "enhance":
        content = Path(args.input).read_text(encoding="utf-8")
        images_dir = args.images_dir or str(Path(args.input).parent)

        analyzer = ImageAnalyzer(
            api_base_url=args.vlm_url,
            dummy_mode=args.dummy,
        )

        enhanced = enhance_markdown(
            markdown=content,
            images_dir=images_dir,
            image_analyzer=analyzer,
        )

        Path(args.output).write_text(enhanced, encoding="utf-8")
        print(f"Saved enhanced Markdown to: {args.output}")

    elif args.md_command == "export":
        input_path = Path(args.input)

        # Check if input is AST JSON
        if input_path.suffix == ".json":
            import json
            with open(input_path, encoding="utf-8") as f:
                tokens = json.load(f)
        else:
            ast_handler = MarkdownAST()
            content = input_path.read_text(encoding="utf-8")
            tokens = ast_handler.parse(content)

        if args.format == "markdown":
            export_to_markdown(tokens, args.output)
        elif args.format == "json":
            export_to_json(tokens, args.output)
        elif args.format == "structured":
            export_to_structured_data(tokens, args.output)

        print(f"Saved {args.format} to: {args.output}")

    return 0


def _run_image(args) -> int:
    """Run image analysis only."""
    analyzer = ImageAnalyzer(
        api_base_url=args.vlm_url,
        dummy_mode=args.dummy,
    )
    return img_main([args.input, args.output, "--dummy" if args.dummy else ""])
