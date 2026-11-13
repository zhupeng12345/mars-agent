"""Compatibility entry point for the F-GMVAE-T training plan."""

from .train_visual import main, run_smoke_training, train_manifest

__all__ = ["main", "run_smoke_training", "train_manifest"]

if __name__ == "__main__":
    raise SystemExit(main())
