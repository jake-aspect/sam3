"""
Tests that verify the package can be imported without pulling in heavy or
platform-specific transitive dependencies (triton, CUDA kernels, etc.).
"""

import importlib
import sys


class TestPackageImport:
    """Verify top-level ``import sam3`` behaviour."""

    def test_import_sam3(self):
        """``import sam3`` must succeed without error."""
        import sam3

        assert hasattr(sam3, "__version__")

    def test_version_is_string(self):
        import sam3

        assert isinstance(sam3.__version__, str)

    def test_triton_not_imported_on_bare_import(self):
        """Importing sam3 must NOT transitively import triton."""
        # Clear any cached import so we get a clean slate
        mods_before = set(sys.modules.keys())
        importlib.invalidate_caches()

        import sam3  # noqa: F401

        triton_modules = {
            k for k in sys.modules.keys() - mods_before if k.startswith("triton")
        }
        assert triton_modules == set(), (
            f"Bare 'import sam3' pulled in triton modules: {triton_modules}"
        )

    def test_lazy_build_function_exists(self):
        """``sam3.build_sam3_image_model`` is accessible via lazy attribute."""
        import sam3

        assert "build_sam3_image_model" in sam3.__all__
