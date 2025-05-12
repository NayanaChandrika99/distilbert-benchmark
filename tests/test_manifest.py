"""
Tests for artifact manifest generation.

This module tests the generation of SHA256 checksums for project artifacts.
"""

import os
import sys
import json
import tempfile
import unittest

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import manifest module (to be created)
from src import manifest


class TestManifestGeneration(unittest.TestCase):
    """Test suite for artifact manifest generation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create some test files
        self.test_files = {}
        for i in range(3):
            file_path = os.path.join(self.temp_dir.name, f"test_file_{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Test content {i}")
            self.test_files[file_path] = f"Test content {i}"

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_generate_file_checksum(self):
        """Test generating a SHA256 checksum for a single file."""
        # Test with the first file
        test_file = list(self.test_files.keys())[0]

        # Generate checksum
        checksum = manifest.generate_file_checksum(test_file)

        # Verify checksum is a 64-character hex string (SHA256)
        self.assertEqual(len(checksum), 64, "SHA256 checksum should be 64 characters")
        self.assertTrue(
            all(c in "0123456789abcdef" for c in checksum),
            "Checksum should only contain hexadecimal characters",
        )

        # Verify checksum is correct and deterministic
        checksum2 = manifest.generate_file_checksum(test_file)
        self.assertEqual(checksum, checksum2, "Checksum should be deterministic")

    def test_generate_manifest(self):
        """Test generating a manifest for a directory."""
        manifest_path = os.path.join(self.temp_dir.name, "manifest.json")

        # Generate manifest
        result = manifest.generate_manifest(
            directory=self.temp_dir.name,
            output_file=manifest_path,
            patterns=["*.txt"],
        )

        # Verify function returned success
        self.assertTrue(result, "Manifest generation should return success")

        # Verify manifest file exists
        self.assertTrue(
            os.path.exists(manifest_path), "Manifest file should be created"
        )

        # Verify manifest content
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)

        # Check manifest structure
        self.assertIn(
            "generated_at", manifest_data, "Manifest should include timestamp"
        )
        self.assertIn("files", manifest_data, "Manifest should include files list")

        # Check that all test files are included
        manifest_files = {
            os.path.basename(f["path"]): f["checksum"] for f in manifest_data["files"]
        }
        for test_file in self.test_files.keys():
            basename = os.path.basename(test_file)
            self.assertIn(
                basename, manifest_files, f"Manifest should include {basename}"
            )

    def test_verify_manifest(self):
        """Test verifying a manifest against actual files."""
        manifest_path = os.path.join(self.temp_dir.name, "manifest.json")

        # Generate manifest
        manifest.generate_manifest(
            directory=self.temp_dir.name,
            output_file=manifest_path,
            patterns=["*.txt"],
        )

        # Verify manifest
        verification_result = manifest.verify_manifest(manifest_path)
        self.assertTrue(verification_result["valid"], "Manifest should be valid")
        self.assertEqual(
            len(verification_result["missing"]), 0, "No files should be missing"
        )
        self.assertEqual(
            len(verification_result["modified"]), 0, "No files should be modified"
        )

        # Modify a file and test verification
        test_file = list(self.test_files.keys())[0]
        with open(test_file, "w") as f:
            f.write("Modified content")

        # Verify manifest with modified file
        verification_result = manifest.verify_manifest(manifest_path)
        self.assertFalse(
            verification_result["valid"],
            "Manifest should be invalid after file modification",
        )
        self.assertEqual(
            len(verification_result["missing"]), 0, "No files should be missing"
        )
        self.assertEqual(
            len(verification_result["modified"]), 1, "One file should be modified"
        )
        self.assertEqual(
            verification_result["modified"][0],
            os.path.basename(test_file),
            "Modified file should be reported",
        )


if __name__ == "__main__":
    unittest.main()
