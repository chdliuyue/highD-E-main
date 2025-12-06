"""Data preprocessing package for highD-E-main."""

__all__ = ["HighDDataBuilder"]


def __getattr__(name):
    if name == "HighDDataBuilder":
        from data_preproc.builder import HighDDataBuilder

        return HighDDataBuilder
    raise AttributeError(f"module 'data_preproc' has no attribute {name!r}")
