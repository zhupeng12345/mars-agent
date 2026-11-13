"""Compatibility wrapper for media manifest generation.

The command implementation lives under :mod:`agentic.scripts`; this module is
kept beside the normalizer so older callers that import
``agentic.media.build_media_manifest`` continue to work.
"""

from agentic.media.normalize import build_media_manifest, manifests_to_jsonl
from agentic.scripts.build_media_manifest import main

__all__ = ["build_media_manifest", "manifests_to_jsonl", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
