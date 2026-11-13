"""CPU smoke and GPU-ready visual training entry points."""

__all__ = ["run_smoke_training", "train_manifest"]


def __getattr__(name: str):
    if name in __all__:
        from .train_visual import run_smoke_training, train_manifest
        return {"run_smoke_training": run_smoke_training, "train_manifest": train_manifest}[name]
    raise AttributeError(name)
