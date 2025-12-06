"""Data preprocessing package for highD-E-main."""

__all__ = ["L1Builder", "HighDDataBuilder"]


def __getattr__(name):
    if name == "L1Builder":
        from data_preproc.l1_builder import L1Builder

        return L1Builder
    if name == "HighDDataBuilder":
        from data_preproc.builder import HighDDataBuilder

        return HighDDataBuilder
    raise AttributeError(f"module 'data_preproc' has no attribute {name!r}")
