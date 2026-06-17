"""Model definitions and training entrypoints.

Keep this package init lightweight. Training scripts import specific model
modules lazily to avoid loading unrelated frameworks under tight memory limits.
"""

__all__: list[str] = []
