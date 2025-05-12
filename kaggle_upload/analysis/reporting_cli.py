#!/usr/bin/env python
"""
Command-line interface for generating reports and slide decks.

This script provides a command-line interface to the reporting module.
"""

import os
import sys
import argparse
import logging
from reporting import generate_html, generate_pdf, generate_slides

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reports and slide decks from benchmark insights"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input markdown file with insights",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output report/slide deck file",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "pdf", "pptx"],
        default="html",
        help="Output format (html, pdf, or pptx)",
    )
    parser.add_argument(
        "--figures",
        type=str,
        help="Path to directory containing figures (required for PPTX format)",
    )
    parser.add_argument(
        "--bibliography",
        action="store_true",
        help="Include bibliography processing for citations",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="DistilBERT Benchmark Results",
        help="Title for slide deck (only used for PPTX format)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Process based on requested format
    if args.format == "html":
        logger.info(f"Generating HTML report from {args.input}...")
        result = generate_html(args.input, args.output)
    elif args.format == "pdf":
        logger.info(f"Generating PDF report from {args.input}...")
        result = generate_pdf(args.input, args.output, args.bibliography)
    elif args.format == "pptx":
        if not args.figures:
            logger.error("Figures directory is required for PPTX format")
            return 1

        if not os.path.exists(args.figures):
            logger.error(f"Figures directory not found: {args.figures}")
            return 1

        logger.info(f"Generating PowerPoint slide deck from {args.input}...")
        result = generate_slides(
            args.input, args.figures, args.output, args.title, args.bibliography
        )
    else:
        logger.error(f"Unsupported format: {args.format}")
        return 1

    if not result:
        logger.error(f"Failed to generate {args.format.upper()} from {args.input}")
        return 1

    logger.info(f"Successfully generated {args.format.upper()} at {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
