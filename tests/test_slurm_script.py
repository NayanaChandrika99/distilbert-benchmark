"""
Tests for SLURM integration scripts.

This module tests the SLURM job templates and scripts for running benchmarks
on HPC clusters.
"""

import os
import re
import sys
import subprocess
import unittest

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestSlurmScript(unittest.TestCase):
    """Test suite for SLURM integration scripts."""

    def setUp(self):
        """Set up the test environment."""
        self.slurm_script_path = os.path.join("cluster", "bench_distilbert.slurm")
        self.cpu_script_path = os.path.join("cluster", "bench_distilbert_cpu.slurm")

        # Ensure the scripts exist
        for script in [self.slurm_script_path, self.cpu_script_path]:
            self.assertTrue(os.path.exists(script), f"Script {script} does not exist")

    def test_slurm_script_syntax(self):
        """Test that the SLURM script has correct syntax."""
        # Test the GPU script
        self._verify_script_syntax(self.slurm_script_path)

        # Test the CPU script
        self._verify_script_syntax(self.cpu_script_path)

    def _verify_script_syntax(self, script_path):
        """Verify the syntax of a SLURM script with shell -n."""
        try:
            result = subprocess.run(
                ["bash", "-n", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            # If subprocess.run() doesn't raise an exception, the script syntax is correct
            self.assertEqual(
                result.returncode, 0, f"Script {script_path} has syntax errors"
            )
        except subprocess.CalledProcessError as e:
            self.fail(f"Script {script_path} has syntax errors: {e.stderr.decode()}")

    def test_required_sbatch_directives(self):
        """Test that the SLURM script has all required SBATCH directives."""
        # Test the GPU script
        required_directives_gpu = [
            "job-name",
            "output",
            "error",
            "time",
            "cpus-per-task",
            "mem",
            "gres=gpu",
            "nodes",
        ]
        self._verify_sbatch_directives(self.slurm_script_path, required_directives_gpu)

        # Test the CPU script
        required_directives_cpu = [
            "job-name",
            "output",
            "error",
            "time",
            "cpus-per-task",
            "mem",
            "nodes",
        ]
        self._verify_sbatch_directives(self.cpu_script_path, required_directives_cpu)

    def _verify_sbatch_directives(self, script_path, required_directives):
        """Verify that a SLURM script has all required SBATCH directives."""
        with open(script_path, "r") as f:
            content = f.read()

        for directive in required_directives:
            # Simplified pattern to match directives regardless of content after the directive
            pattern = r"#SBATCH\s+--?" + re.escape(directive)
            self.assertTrue(
                re.search(pattern, content),
                f"Directive '{directive}' not found in {script_path}",
            )

    def test_parameterizable_resources(self):
        """Test that the SLURM script can be parameterized for different resource requirements."""
        for script_path in [self.slurm_script_path, self.cpu_script_path]:
            with open(script_path, "r") as f:
                content = f.read()

            # Check for environment variable usage for parameterization
            self.assertTrue(
                "${" in content or "$(" in content,
                f"Script {script_path} does not use environment variables for parameterization",
            )

    def test_module_loading(self):
        """Test that the SLURM script loads required modules."""
        for script_path in [self.slurm_script_path, self.cpu_script_path]:
            with open(script_path, "r") as f:
                content = f.read()

            # Check for module loading commands
            self.assertTrue(
                "module load" in content,
                f"Script {script_path} does not load required modules",
            )

    def test_conda_environment_activation(self):
        """Test that the SLURM script activates the conda environment."""
        for script_path in [self.slurm_script_path, self.cpu_script_path]:
            with open(script_path, "r") as f:
                content = f.read()

            # Check for conda environment activation
            self.assertTrue(
                ("source activate" in content or "conda activate" in content),
                f"Script {script_path} does not activate the conda environment",
            )

    def test_rsync_functionality(self):
        """Test that the SLURM script includes rsync functionality for result transfer."""
        for script_path in [self.slurm_script_path, self.cpu_script_path]:
            with open(script_path, "r") as f:
                content = f.read()

            # Check for rsync command
            self.assertTrue(
                "rsync" in content,
                f"Script {script_path} does not include rsync functionality",
            )

    def test_benchmark_command(self):
        """Test that the SLURM script runs the benchmark command."""
        for script_path in [self.slurm_script_path, self.cpu_script_path]:
            with open(script_path, "r") as f:
                content = f.read()

            # Check for benchmark command
            self.assertTrue(
                "python src/runner.py" in content,
                f"Script {script_path} does not run the benchmark command",
            )

    def test_cpu_gpu_differentiation(self):
        """Test that the CPU and GPU scripts are appropriately differentiated."""
        # Read both scripts
        with open(self.slurm_script_path, "r") as f:
            gpu_content = f.read()
        with open(self.cpu_script_path, "r") as f:
            cpu_content = f.read()

        # GPU script should have GPU-specific settings
        self.assertTrue(
            "#SBATCH --gres=gpu" in gpu_content,
            "GPU script does not have GPU resource allocation",
        )

        # CPU script should not have GPU-specific settings
        self.assertFalse(
            "#SBATCH --gres=gpu" in cpu_content,
            "CPU script incorrectly has GPU resource allocation",
        )

        # Check device setting in benchmark command - updated patterns to handle variables
        self.assertTrue(
            ("--device cuda" in gpu_content)
            or ("--device gpu" in gpu_content)
            or ("--device ${DEVICE:-cuda}" in gpu_content),
            "GPU script does not set GPU device for benchmark",
        )
        self.assertTrue(
            "--device cpu" in cpu_content,
            "CPU script does not set CPU device for benchmark",
        )


if __name__ == "__main__":
    unittest.main()
