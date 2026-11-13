"""Compatibility entry point for ``python -m agentic.build_media_manifest``."""

from agentic.scripts.build_media_manifest import main


if __name__ == "__main__":
    raise SystemExit(main())
