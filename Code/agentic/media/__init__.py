"""Media normalization helpers used by inspection agents.

The module intentionally keeps media handling independent of the LangGraph
workflow.  It can therefore be used by dataset builders, API handlers and
agents that only need a JSON-safe manifest.
"""

from .normalize import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MediaNormalizer,
    build_media_manifest,
    normalize_media,
    read_image,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "MediaNormalizer",
    "build_media_manifest",
    "normalize_media",
    "read_image",
]
