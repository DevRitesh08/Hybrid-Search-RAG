"""Package init — exposes version and settings."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hybrid-rag-hf-docs")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
