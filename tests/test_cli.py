"""Smoke tests for the SAM3 command-line interface."""

import subprocess
import sys


class TestCLI:
    """Verify the CLI entry point is wired up correctly."""

    def test_help_exits_zero(self):
        """``sam3 --help`` should succeed."""
        result = subprocess.run(
            [sys.executable, "-m", "sam3.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "SAM3" in result.stdout

    def test_version_flag(self):
        """``sam3 --version`` should print the version."""
        result = subprocess.run(
            [sys.executable, "-m", "sam3.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "sam3" in result.stdout

    def test_no_args_prints_help(self):
        """Running with no arguments should print help and exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "sam3.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "segment" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_segment_subcommand_exists(self):
        """The 'segment' subcommand should be registered."""
        result = subprocess.run(
            [sys.executable, "-m", "sam3.cli", "segment"],
            capture_output=True,
            text=True,
        )
        # For now it's a stub, so it should return 1 with a message
        assert "not yet implemented" in result.stdout.lower() or result.returncode in (
            0,
            1,
        )
