"""
Generate and verify SHA256 manifests for project artifacts.

This module provides functionality to create and verify checksums for project files.
"""

import os
import sys
import json
import hashlib
import datetime
import logging
import glob
from typing import Dict, List, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)


def generate_file_checksum(file_path: str) -> str:
    """
    Generate a SHA256 checksum for a file.

    Args:
        file_path: Path to the file

    Returns:
        SHA256 checksum as a hex string
    """
    hash_sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read and update hash in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def generate_manifest(
    directory: str,
    output_file: str,
    patterns: List[str] = ["*.py", "*.md", "*.txt", "*.html", "*.pdf", "*.pptx"],
    exclude_patterns: List[str] = ["__pycache__/*", "*.pyc", "*.git/*"],
) -> bool:
    """
    Generate a manifest file with SHA256 checksums for all files matching patterns.

    Args:
        directory: Root directory to scan for files
        output_file: Path to write the manifest JSON file
        patterns: List of glob patterns to include (default: common file types)
        exclude_patterns: List of glob patterns to exclude (default: common exclusions)

    Returns:
        True if manifest was successfully created, False otherwise
    """
    try:
        # Ensure directory is absolute
        directory = os.path.abspath(directory)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        # Initialize manifest structure
        manifest = {"generated_at": datetime.datetime.now().isoformat(), "files": []}

        # Process each pattern
        all_files = set()

        for pattern in patterns:
            # Handle both absolute patterns and patterns relative to directory
            if os.path.isabs(pattern):
                matched_files = glob.glob(pattern, recursive=True)
            else:
                matched_files = glob.glob(
                    os.path.join(directory, "**", pattern), recursive=True
                )

            all_files.update(matched_files)

        # Apply exclude patterns
        for exclude in exclude_patterns:
            if os.path.isabs(exclude):
                excluded = set(glob.glob(exclude, recursive=True))
            else:
                excluded = set(
                    glob.glob(os.path.join(directory, "**", exclude), recursive=True)
                )

            all_files -= excluded

        # Generate checksums for each file
        for file_path in sorted(all_files):
            if os.path.isfile(file_path):
                try:
                    rel_path = os.path.relpath(file_path, directory)
                    checksum = generate_file_checksum(file_path)

                    manifest["files"].append(
                        {
                            "path": rel_path,
                            "checksum": checksum,
                            "size": os.path.getsize(file_path),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")

        # Write manifest to file
        with open(output_file, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Generated manifest with {len(manifest['files'])} files at {output_file}"
        )
        return True

    except Exception as e:
        logger.error(f"Error generating manifest: {e}")
        return False


def verify_manifest(
    manifest_file: str, base_directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify files against their checksums in a manifest file.

    Args:
        manifest_file: Path to the manifest JSON file
        base_directory: Base directory for resolving relative paths (default: manifest file directory)

    Returns:
        Dict with verification results:
            - valid: True if all files match, False otherwise
            - missing: List of files in manifest that are missing
            - modified: List of files that have different checksums
            - total: Total number of files checked
    """
    try:
        # Determine base directory
        if base_directory is None:
            base_directory = os.path.dirname(os.path.abspath(manifest_file))

        # Read manifest
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        # Initialize verification results
        verification = {
            "valid": True,
            "missing": [],
            "modified": [],
            "total": len(manifest["files"]),
        }

        # Verify each file
        for file_entry in manifest["files"]:
            file_path = os.path.join(base_directory, file_entry["path"])

            if not os.path.exists(file_path):
                verification["missing"].append(file_entry["path"])
                verification["valid"] = False
                continue

            current_checksum = generate_file_checksum(file_path)

            if current_checksum != file_entry["checksum"]:
                verification["modified"].append(file_entry["path"])
                verification["valid"] = False

        return verification

    except Exception as e:
        logger.error(f"Error verifying manifest: {e}")
        return {
            "valid": False,
            "error": str(e),
            "missing": [],
            "modified": [],
            "total": 0,
        }


def main():
    """Command-line interface for manifest generation and verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and verify file manifests with SHA256 checksums"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Generate manifest command
    generate_parser = subparsers.add_parser("generate", help="Generate a manifest file")
    generate_parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Directory to scan for files (default: current directory)",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="manifest.json",
        help="Output manifest file path (default: manifest.json)",
    )
    generate_parser.add_argument(
        "--patterns",
        "-p",
        type=str,
        nargs="+",
        default=["*.py", "*.md", "*.txt", "*.html", "*.pdf", "*.pptx"],
        help="File patterns to include (default: common file types)",
    )
    generate_parser.add_argument(
        "--exclude",
        "-e",
        type=str,
        nargs="+",
        default=["__pycache__/*", "*.pyc", ".git/*"],
        help="File patterns to exclude (default: common exclusions)",
    )

    # Verify manifest command
    verify_parser = subparsers.add_parser(
        "verify", help="Verify files against a manifest"
    )
    verify_parser.add_argument(
        "--manifest", "-m", type=str, required=True, help="Path to manifest file"
    )
    verify_parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=None,
        help="Base directory for resolving paths (default: manifest file directory)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "generate":
        result = generate_manifest(
            directory=args.directory,
            output_file=args.output,
            patterns=args.patterns,
            exclude_patterns=args.exclude,
        )
        sys.exit(0 if result else 1)

    elif args.command == "verify":
        result = verify_manifest(
            manifest_file=args.manifest, base_directory=args.directory
        )

        if result["valid"]:
            logger.info(
                f"Manifest verification successful. All {result['total']} files match."
            )
            sys.exit(0)
        else:
            logger.error("Manifest verification failed:")
            if result["missing"]:
                logger.error(
                    f"- Missing files ({len(result['missing'])}): {', '.join(result['missing'][:5])}"
                )
                if len(result["missing"]) > 5:
                    logger.error(f"  ... and {len(result['missing']) - 5} more")

            if result["modified"]:
                logger.error(
                    f"- Modified files ({len(result['modified'])}): {', '.join(result['modified'][:5])}"
                )
                if len(result["modified"]) > 5:
                    logger.error(f"  ... and {len(result['modified']) - 5} more")

            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
