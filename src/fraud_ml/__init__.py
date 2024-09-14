"""Fraud risk scoring and decision support for synthetic financial transactions.

Educational implementation. Not intended for production fraud decisioning, regulatory
compliance or autonomous blocking of transactions.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fraud-detection-ml")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
