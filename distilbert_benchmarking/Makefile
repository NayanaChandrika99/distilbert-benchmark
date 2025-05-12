.PHONY: setup smoke benchmark benchmark-gpu list-devices analyse analyse-plots analyse-insights analyse-compare release clean test-analysis report report-pdf report-pptx report-full quick-report manifest verify-manifest setup-pre-commit lint test benchmark-cpu benchmark-gpu benchmark-quick benchmark-all

# Configuration
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
RESULTS_DIR := results-$(TIMESTAMP)
CONFIG_FILE := config.yaml

# Default target
all: benchmark analyse report

# Setup environment
setup:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml
	@echo "Environment 'distilbert-benchmark' created. Activate with 'conda activate distilbert-benchmark'"

# Setup pre-commit hooks
setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed. Run 'pre-commit run --all-files' to check existing files."

# Smoke test to validate setup
smoke:
	@echo "Running smoke test..."
	python -c "import torch; import transformers; import datasets; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Datasets:', datasets.__version__)"
	python src/runner.py --smoke-test --output smoke-test-results.jsonl
	@echo "Smoke test completed successfully!"

# List available devices
list-devices:
	@echo "Listing available devices..."
	python src/runner.py --list-devices

# Run benchmarks on CPU
benchmark:
	@echo "Running CPU benchmarks..."
	@mkdir -p $(RESULTS_DIR)
	python src/runner.py --config $(CONFIG_FILE) --device cpu --output $(RESULTS_DIR)/cpu-results.jsonl
	@echo "CPU benchmark results saved to $(RESULTS_DIR)/cpu-results.jsonl"

# Run benchmarks on GPU if available
benchmark-gpu:
	@echo "Running GPU benchmarks..."
	@mkdir -p $(RESULTS_DIR)
	python src/runner.py --config $(CONFIG_FILE) --device cuda --output $(RESULTS_DIR)/gpu-results.jsonl
	@echo "GPU benchmark results saved to $(RESULTS_DIR)/gpu-results.jsonl"

# Run benchmarks on both CPU and GPU with comparisons
benchmark-all: benchmark benchmark-gpu
	@echo "Running comparison analysis..."
	python analysis/compare_results.py --cpu $(RESULTS_DIR)/cpu-results.jsonl --gpu $(RESULTS_DIR)/gpu-results.jsonl --output $(RESULTS_DIR)/comparison.md
	@echo "All benchmarks completed. Comparison saved to $(RESULTS_DIR)/comparison.md"

# Generate only plots from existing results (specify RESULTS_DIR)
analyse-plots:
	@echo "Generating plots from benchmark results..."
	@mkdir -p $(RESULTS_DIR)/figures
	python analysis/plot_metrics.py --input $(RESULTS_DIR)/cpu-results.jsonl --output $(RESULTS_DIR)/figures
	@if [ -f "$(RESULTS_DIR)/gpu-results.jsonl" ]; then \
		python analysis/plot_metrics.py --input $(RESULTS_DIR)/gpu-results.jsonl --output $(RESULTS_DIR)/figures/gpu; \
		python analysis/plot_metrics.py --input $(RESULTS_DIR)/cpu-results.jsonl --compare $(RESULTS_DIR)/gpu-results.jsonl --output $(RESULTS_DIR)/figures/comparison; \
	fi
	@echo "Plots generated in $(RESULTS_DIR)/figures"

# Generate only insights report from existing results (specify RESULTS_DIR)
analyse-insights:
	@echo "Generating insights from benchmark results..."
	@mkdir -p $(RESULTS_DIR)
	python analysis/generate_report.py --input $(RESULTS_DIR)/cpu-results.jsonl --output $(RESULTS_DIR)/cpu-insights.md --figures $(RESULTS_DIR)/figures
	@if [ -f "$(RESULTS_DIR)/gpu-results.jsonl" ]; then \
		python analysis/generate_report.py --input $(RESULTS_DIR)/gpu-results.jsonl --output $(RESULTS_DIR)/gpu-insights.md --figures $(RESULTS_DIR)/figures/gpu; \
	fi
	@echo "Insights reports generated in $(RESULTS_DIR)"

# Generate CPU vs GPU comparison from existing results (specify RESULTS_DIR)
analyse-compare:
	@echo "Generating CPU vs GPU comparison..."
	@if [ -f "$(RESULTS_DIR)/cpu-results.jsonl" ] && [ -f "$(RESULTS_DIR)/gpu-results.jsonl" ]; then \
		python analysis/compare_results.py --cpu $(RESULTS_DIR)/cpu-results.jsonl --gpu $(RESULTS_DIR)/gpu-results.jsonl --output $(RESULTS_DIR)/comparison.md; \
		echo "Comparison saved to $(RESULTS_DIR)/comparison.md"; \
	else \
		echo "Error: Both CPU and GPU results are required for comparison"; \
		exit 1; \
	fi

# Analyze results - runs all analysis scripts
analyse: analyse-plots analyse-insights analyse-compare
	@echo "Analysis completed. Results in $(RESULTS_DIR)"

# Test analysis scripts
test-analysis:
	@echo "Testing analysis components..."
	python tests/test_analysis.py
	@echo "Analysis tests completed successfully!"

# Test reporting functionality
test-reporting:
	@echo "Testing reporting components..."
	python tests/test_reporting.py
	@echo "Reporting tests completed successfully!"

# Generate HTML reports
report:
	@echo "Generating HTML reports..."
	@mkdir -p $(RESULTS_DIR)
	python analysis/reporting.py --input $(RESULTS_DIR)/cpu-insights.md --output $(RESULTS_DIR)/cpu-report.html --format html
	@if [ -f "$(RESULTS_DIR)/gpu-insights.md" ]; then \
		python analysis/reporting.py --input $(RESULTS_DIR)/gpu-insights.md --output $(RESULTS_DIR)/gpu-report.html --format html; \
	fi
	@if [ -f "$(RESULTS_DIR)/comparison.md" ]; then \
		python analysis/reporting.py --input $(RESULTS_DIR)/comparison.md --output $(RESULTS_DIR)/comparison-report.html --format html; \
	fi
	@echo "HTML report generation completed. See HTML reports in $(RESULTS_DIR)"

# Generate PDF reports
report-pdf:
	@echo "Generating PDF reports..."
	@mkdir -p $(RESULTS_DIR)
	python analysis/reporting.py --input $(RESULTS_DIR)/cpu-insights.md --output $(RESULTS_DIR)/cpu-report.pdf --format pdf --bibliography
	@if [ -f "$(RESULTS_DIR)/gpu-insights.md" ]; then \
		python analysis/reporting.py --input $(RESULTS_DIR)/gpu-insights.md --output $(RESULTS_DIR)/gpu-report.pdf --format pdf --bibliography; \
	fi
	@if [ -f "$(RESULTS_DIR)/comparison.md" ]; then \
		python analysis/reporting.py --input $(RESULTS_DIR)/comparison.md --output $(RESULTS_DIR)/comparison-report.pdf --format pdf --bibliography; \
	fi
	@echo "PDF report generation completed. See PDF files in $(RESULTS_DIR)"

# Generate PowerPoint slide decks
report-pptx:
	@echo "Generating PowerPoint slide decks..."
	@mkdir -p $(RESULTS_DIR)
	python analysis/reporting.py --input $(RESULTS_DIR)/cpu-insights.md --output $(RESULTS_DIR)/cpu-slides.pptx --format pptx --figures $(RESULTS_DIR)/figures --bibliography
	@if [ -f "$(RESULTS_DIR)/gpu-insights.md" ]; then \
		python analysis/reporting.py --input $(RESULTS_DIR)/gpu-insights.md --output $(RESULTS_DIR)/gpu-slides.pptx --format pptx --figures $(RESULTS_DIR)/figures/gpu --bibliography; \
	fi
	@if [ -f "$(RESULTS_DIR)/comparison.md" ]; then \
		python analysis/reporting.py --input $(RESULTS_DIR)/comparison.md --output $(RESULTS_DIR)/comparison-slides.pptx --format pptx --figures $(RESULTS_DIR)/figures/comparison --bibliography; \
	fi
	@echo "PowerPoint slide deck generation completed. See PPTX files in $(RESULTS_DIR)"

# Generate all report formats (HTML, PDF, PPTX)
report-full: report report-pdf report-pptx
	@echo "All report formats generated in $(RESULTS_DIR)"

# Quick report - generate plots and insights from latest results
quick-report:
	@echo "Generating quick report from smoke test results..."
	@rm -rf quick-report
	@mkdir -p quick-report/figures
	python analysis/plot_metrics.py --input smoke-test-results.jsonl --output quick-report/figures
	python analysis/generate_report.py --input smoke-test-results.jsonl --output quick-report/insights.md --figures quick-report/figures
	python analysis/reporting.py --input quick-report/insights.md --output quick-report/report.html --format html
	python analysis/reporting.py --input quick-report/insights.md --output quick-report/slides.pptx --format pptx --figures quick-report/figures
	@echo "Quick report generated in quick-report directory"

# Generate SHA256 manifest for project artifacts
manifest:
	@echo "Generating SHA256 manifest for project artifacts..."
	@mkdir -p release
	python src/manifest.py generate --directory . --output release/manifest.json --patterns "*.py" "*.md" "*.yml" "*.yaml" "*.json" "Makefile" "*.html" "*.pdf" "*.pptx" --exclude "__pycache__/*" "*.pyc" ".git/*" ".pytest_cache/*" "*.egg-info/*"
	@echo "Manifest generated at release/manifest.json"

# Verify SHA256 manifest for project artifacts
verify-manifest:
	@echo "Verifying SHA256 manifest for project artifacts..."
	python src/manifest.py verify --manifest release/manifest.json
	@echo "Manifest verification completed"

# Run code quality checks (lint, format, type check)
lint:
	@echo "Running code quality checks..."
	pip install black flake8 mypy ruff
	flake8 src/ analysis/ tests/
	black --check src/ analysis/ tests/
	mypy src/ analysis/
	ruff src/ analysis/ tests/
	@echo "Code quality checks completed"

# Run all tests with coverage report
test:
	@echo "Running all tests with coverage..."
	pytest --cov=src --cov=analysis --cov-report=term --cov-report=html tests/
	@echo "Test coverage report generated in htmlcov/"

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup completed!"

# SLURM submission
slurm:
	@echo "Submitting job to SLURM..."
	./cluster/submit_benchmark.sh
	@echo "Job submitted to SLURM queue."

# Add specialized SLURM targets for CPU and GPU
slurm-cpu:
	@echo "Submitting CPU job to SLURM..."
	./cluster/submit_benchmark.sh --device cpu
	@echo "CPU job submitted to SLURM queue."

slurm-gpu:
	@echo "Submitting GPU job to SLURM..."
	./cluster/submit_benchmark.sh --device gpu
	@echo "GPU job submitted to SLURM queue."

# Add a target for submitting full benchmark suite with analysis
slurm-full:
	@echo "Submitting full benchmark suite to SLURM with analysis..."
	./cluster/submit_benchmark.sh --device both --analyze
	@echo "Benchmark suite submitted to SLURM queue."

# Add mock SLURM environment target
mock-slurm:
	@echo "Setting up mock SLURM environment for testing..."
	./cluster/mock_slurm_env.sh

# Add target for testing SLURM scripts
test-slurm: mock-slurm
	@echo "Testing SLURM scripts using mock environment..."
	python tests/test_slurm_script.py
	@echo "SLURM script tests completed successfully!"

# Run all tests
test-all: test test-analysis test-reporting test-slurm
	@echo "All tests completed successfully!"

# Release target for final deliverables
release: benchmark-all analyse report-full manifest
	@echo "Creating release package..."
	@mkdir -p release
	cp $(RESULTS_DIR)/*.html release/
	cp $(RESULTS_DIR)/*.pdf release/ 2>/dev/null || true
	cp $(RESULTS_DIR)/*.pptx release/ 2>/dev/null || true
	cp -r $(RESULTS_DIR)/figures release/
	python src/manifest.py generate --directory release --output release/manifest.json --patterns "*.html" "*.pdf" "*.pptx" "*.md" "*.json" "figures/**/*.png" "figures/**/*.jpg"
	@echo "Release package created in 'release' directory with SHA256 manifest"

# Help command
help:
	@echo "Makefile targets:"
	@echo "  setup            - Create conda environment"
	@echo "  setup-pre-commit - Install pre-commit hooks for code quality"
	@echo "  smoke            - Run smoke test to verify environment"
	@echo "  list-devices     - List available CPU and GPU devices"
	@echo "  benchmark        - Run benchmarks on CPU and save results"
	@echo "  benchmark-gpu    - Run benchmarks on GPU (if available) and save results"
	@echo "  benchmark-all    - Run benchmarks on both CPU and GPU with comparison"
	@echo "  analyse          - Run all analysis scripts (plots, insights, comparison)"
	@echo "  analyse-plots    - Generate only plots from benchmark results"
	@echo "  analyse-insights - Generate only insights report from benchmark results"
	@echo "  analyse-compare  - Generate CPU vs GPU comparison from benchmark results"
	@echo "  report           - Create HTML reports from insights"
	@echo "  report-pdf       - Create PDF reports from insights"
	@echo "  report-pptx      - Create PowerPoint slide decks from insights"
	@echo "  report-full      - Generate all report formats (HTML, PDF, PPTX)"
	@echo "  quick-report     - Generate plots and report from smoke test results"
	@echo "  manifest         - Generate SHA256 manifest for project artifacts"
	@echo "  verify-manifest  - Verify project artifacts against manifest"
	@echo "  lint             - Run code quality checks (flake8, black, mypy)"
	@echo "  test             - Run tests with coverage report"
	@echo "  test-analysis    - Run tests for analysis components"
	@echo "  test-reporting   - Run tests for reporting components"
	@echo "  test-slurm       - Test SLURM scripts using mock environment"
	@echo "  test-all         - Run all tests"
	@echo "  release          - Create a release package with all reports, figures, and manifest"
	@echo "  slurm            - Submit job to SLURM cluster"
	@echo "  slurm-cpu        - Submit CPU job to SLURM"
	@echo "  slurm-gpu        - Submit GPU job to SLURM"
	@echo "  slurm-full       - Submit full benchmark suite to SLURM with analysis"
	@echo "  mock-slurm       - Set up mock SLURM environment for testing"
	@echo "  clean            - Remove temporary files and caches"
	@echo "  help             - Display this help message"

# Benchmark targets
benchmark-cpu:
	@echo "Running CPU benchmarks..."
	python src/runner.py --device=cpu --output=results/cpu_benchmark.jsonl --batch-sizes=1,2,4,8,16,32,64

benchmark-gpu:
	@echo "Running GPU benchmarks..."
	python src/runner.py --device=cuda:0 --output=results/gpu_benchmark.jsonl --batch-sizes=1,2,4,8,16,32,64 --mixed-precision

benchmark-quick:
	@echo "Running quick benchmark for testing..."
	python src/runner.py --device=cpu --output=results/quick_benchmark.jsonl --batch-sizes=1,2,4 --warmup-runs=1 --iterations=3
