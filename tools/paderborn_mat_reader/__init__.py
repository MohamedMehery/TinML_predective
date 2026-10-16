"""Isolated Paderborn (KAt) MATLAB 5.0 reader.

This package is intentionally NOT integrated into the training or
preprocessing pipelines. It only reads and inspects local .mat files.
"""
from .reader import (
    MatReaderError,
    read_mat,
    inspect_mat,
    ChannelData,
)

__all__ = ["MatReaderError", "read_mat", "inspect_mat", "ChannelData"]
__version__ = "0.1.0"
