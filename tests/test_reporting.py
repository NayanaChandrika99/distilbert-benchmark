"""
Tests for our report generation tools.

Makes sure we can convert benchmark results into pretty PDFs, HTML, and slides.
"""

import os
import sys
import tempfile
import unittest

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import reporting module (to be created)
from analysis import reporting


class TestReportingPipeline(unittest.TestCase):
    """Tests for our fancy report generators"""

    def setUp(self):
        """Create some test data - temp files and dummy markdown"""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a temp markdown file for testing
        self.test_md_file = os.path.join(self.temp_dir.name, "test-insights.md")
        with open(self.test_md_file, "w") as f:
            f.write("# Test Report\n\n")
            f.write("## Introduction\n\n")
            f.write("This is a test report with some sample content.\n\n")
            f.write("## Results\n\n")
            f.write("Here are some results:\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write("| Latency | 50 ms |\n")
            f.write("| Throughput | 100 samples/sec |\n\n")
            f.write("## References\n\n")
            f.write(
                "[@huggingface2023distilbert]: Hugging Face. (2023). DistilBERT. https://huggingface.co/docs/transformers/model_doc/distilbert\n"
            )
            f.write(
                "[@sanh2019distilbert]: Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.\n"
            )

        # Create a sample figure for inclusion
        self.figures_dir = os.path.join(self.temp_dir.name, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        # We'll use a text file to simulate a figure since we don't need to test matplotlib here
        with open(os.path.join(self.figures_dir, "sample_plot.png"), "w") as f:
            f.write("test figure content")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_pdf_generation(self):
        """Check we can turn markdown into a PDF"""
        output_pdf = os.path.join(self.temp_dir.name, "test-report.pdf")

        # Test PDF generation
        result = reporting.generate_pdf(
            input_file=self.test_md_file,
            output_file=output_pdf,
            include_bibliography=True,
        )

        # PDF generation might not work in all environments, so we'll skip file existence check
        # and just verify the function returns success status
        self.assertTrue(result, "PDF generation should return success status")
        self.assertTrue(os.path.exists(output_pdf), "PDF file should be created")

    def test_html_generation(self):
        """Check we can turn markdown into a decent-looking webpage"""
        output_html = os.path.join(self.temp_dir.name, "test-report.html")

        # Test HTML generation
        result = reporting.generate_html(
            input_file=self.test_md_file, output_file=output_html, self_contained=True
        )

        # Check if the HTML file was created
        self.assertTrue(os.path.exists(output_html), "HTML file should be created")
        self.assertTrue(result, "HTML generation should return success status")

        # Verify content
        with open(output_html, "r") as f:
            content = f.read()
            self.assertIn(
                "<h1>Test Report</h1>", content, "HTML should contain the title"
            )
            self.assertIn(
                "<h2>Results</h2>", content, "HTML should contain the Results section"
            )
            self.assertIn("<table>", content, "HTML should contain the table")

    def test_pptx_generation(self):
        """Check we can build a PowerPoint from benchmark results"""
        output_pptx = os.path.join(self.temp_dir.name, "test-slides.pptx")

        # Test PowerPoint generation
        result = reporting.generate_slides(
            markdown_file=self.test_md_file,
            figures_dir=self.figures_dir,
            output_file=output_pptx,
            title="DistilBERT Benchmark Results",
            include_bibliography=True,
        )

        # Check if the PPTX file was created
        self.assertTrue(
            os.path.exists(output_pptx), "PowerPoint file should be created"
        )
        self.assertTrue(result, "PowerPoint generation should return success status")

        # We can't easily check the content of a binary PPTX file in a unit test,
        # so we'll rely on manual testing to verify the content

    def test_bibliography_handling(self):
        """Check we can pull out and format citations properly"""
        # Extract citations from markdown
        citations = reporting.extract_citations(self.test_md_file)

        # Check if citations were extracted correctly
        self.assertIn(
            "huggingface2023distilbert",
            citations,
            "Should extract the Hugging Face citation",
        )
        self.assertIn(
            "sanh2019distilbert", citations, "Should extract the Sanh et al. citation"
        )

        # Test formatting references in APA style
        formatted_refs = reporting.format_references(citations, style="apa")

        # Check formatted references
        self.assertIn(
            "Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019)",
            formatted_refs,
            "Should contain formatted Sanh et al. reference",
        )


class MemoryMetricCollector:
    def __init__(
        self, device: str = "cpu", interval_ms: int = 10, track_gpu: bool = True
    ):
        """Track memory usage over time
        
        Uses a background thread to sample memory every [interval] ms.
        Can track both CPU (RSS) and GPU memory.
        """
        # Rest of the code...

    def _collect_metrics(self):
        """Background thread that polls memory usage
        
        Runs until explicitly stopped via stop_collection()
        """
        # Rest of the code...

def measure_peak_memory_usage(func):
    """Decorator to measure memory used by a function
    
    Usage:
        @measure_peak_memory_usage
        def my_memory_hungry_func():
            # do stuff
    
    Returns the function result plus memory stats.
    """
    # Rest of the code...


if __name__ == "__main__":
    unittest.main()
