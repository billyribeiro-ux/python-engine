"""Scanner framework and concrete scanners.

The ``Scanner`` protocol + ``scanner(name)`` decorator + ``registry``
singleton form a tiny plugin system. New scanners drop into the registry
just by being imported and decorated.
"""

from .base import Scanner, registry, scanner

__all__ = ["Scanner", "registry", "scanner"]
