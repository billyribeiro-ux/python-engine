"""Scanner framework and concrete scanners.

The ``Scanner`` protocol + ``scanner(name)`` decorator + ``registry``
singleton form a tiny plugin system. New scanners drop into the registry
just by being imported and decorated.
"""

from .base import Scanner, registry, scanner
from .concrete import RealisedVolRankScanner, ZScoreMomentumScanner

__all__ = [
    "RealisedVolRankScanner",
    "Scanner",
    "ZScoreMomentumScanner",
    "registry",
    "scanner",
]
