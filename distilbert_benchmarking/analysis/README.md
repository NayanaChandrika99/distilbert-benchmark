# Reporting Pipeline Documentation

This directory contains tools for analyzing benchmark results and generating reports.

## Overview

The reporting pipeline takes markdown insights files generated from benchmark results and produces:

1. HTML reports
2. PDF reports (requires Pandoc)
3. PowerPoint slide decks

## Key Components

- `reporting.py`: Core module for report generation
- `reporting_cli.py`: Command-line interface for reporting module
- `plot_metrics.py`: Generates visualizations from benchmark results
- `generate_report.py`: Creates markdown insights from benchmark data
- `compare_results.py`: Compares CPU and GPU benchmark results

## Usage

### Command Line Interface

The reporting pipeline can be used through the CLI script:

```bash
# Generate HTML report
./reporting_cli.py --input results/cpu-insights.md --output results/cpu-report.html --format html

# Generate PDF report with bibliography
./reporting_cli.py --input results/cpu-insights.md --output results/cpu-report.pdf --format pdf --bibliography

# Generate PowerPoint slide deck
./reporting_cli.py --input results/cpu-insights.md --output results/cpu-slides.pptx --format pptx --figures results/figures --bibliography
```

### Makefile Integration

Several make targets are available for report generation:

- `make report`: Generate HTML reports
- `make report-pdf`: Generate PDF reports
- `make report-pptx`: Generate PowerPoint slide decks
- `make report-full`: Generate all report formats
- `make quick-report`: Create a quick report from smoke test results

## Citation Support

The reporting pipeline supports academic citations in markdown files using the `@citation-key` syntax. Citations are automatically extracted and formatted according to APA style for inclusion in reports and slide decks.

Example citation in markdown:
```markdown
The DistilBERT model [@sanh2019distilbert] is a smaller version of BERT.

## References

[@sanh2019distilbert]: Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
```

## Dependencies

- Python 3.10+
- markdown: For HTML processing
- python-pptx: For PowerPoint generation
- pandoc: For PDF generation (optional, system dependency)

## Testing

Tests for the reporting pipeline are in `tests/test_reporting.py`. Run with:

```bash
python tests/test_reporting.py
# or
make test-reporting
```
