"""Spatial blur map — a vectorised blur_detector.detectBlur.

blur_detector computes a DCT of every patch at every grid point in a Python
double loop; on a 1280-px frame that is ~68k points x 4 scales and cost
~10 s of CPU per photo, three quarters of Stage 1. This module performs the
same arithmetic one grid row at a time, keeping the library's
(D @ patch) @ D.T product order and its per-point argpartition, so the map
is bitwise identical to the library's (see tests/stage1/test_blur_map.py).
The argpartition output order decides which column of the layer matrix a
value lands in, and so its column-max normalisation; any other selection
changes the map.
"""

from __future__ import annotations

import blur_detector
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import BaseModel, ConfigDict

DOWNSAMPLING_FACTOR: int = 4
NUM_SCALES: int = 4
SCALE_START: int = 3
PRE_BLUR_KSIZE: tuple[int, int] = (3, 3)
PRE_BLUR_SIGMA: float = 0.5
GUIDE_BLUR_SIGMA: float = 1.0


class _ScaleKernel(BaseModel):
    """DCT matrix and high-frequency coefficient index for one patch size."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    size: int
    dct: np.ndarray
    high_freq_index: tuple[np.ndarray, np.ndarray]


class _Grid(BaseModel):
    """Padded gradient image and the patch-centre rows/columns sampled on it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    padded: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    kernels: list[_ScaleKernel]


def _scale_sizes() -> list[int]:
    """Return patch sizes 7, 15, 31, 63 (blur_detector.createScalePyramid)."""
    return [(2 ** (SCALE_START + i)) - 1 for i in range(NUM_SCALES)]


def _frequency_band(size: int) -> np.ndarray:
    """Return blur_detector's frequency-band label matrix for one patch size."""
    labels = np.zeros((size, size))
    for i in range(size):
        labels[0: max(0, int(((size - 1) / 2) - i + 1)), i] = 1
    for i in range(size):
        if (size - ((size - 1) / 2) - i) <= 0:
            labels[0: size - i - 1, i] = 2
        else:
            labels[int(size - ((size - 1) / 2) - i - 1): int(size - i - 1), i] = 2
    labels[0, 0] = 3
    return labels


def _dct_matrix(size: int) -> np.ndarray:
    """Return blur_detector's orthonormal DCT-II matrix for one patch size."""
    mesh_cols, mesh_rows = np.meshgrid(np.linspace(0, size - 1, size), np.linspace(0, size - 1, size))
    matrix = np.sqrt(2 / size) * np.cos(np.pi * np.multiply((2 * mesh_cols + 1), mesh_rows) / (2 * size))
    matrix[0, :] = matrix[0, :] / np.sqrt(2)
    return matrix


def _scale_kernel(size: int) -> _ScaleKernel:
    """Build the DCT matrix and high-frequency index for one patch size."""
    return _ScaleKernel(
        size=size,
        dct=_dct_matrix(size),
        high_freq_index=np.where(_frequency_band(size) == 0),
    )


def _build_grid(gradient: np.ndarray) -> _Grid:
    """Zero-pad the gradient image and list the sampled patch centres."""
    sizes = _scale_sizes()
    half = int(max(sizes) / 2)
    padded = np.pad(gradient, half, mode="constant")
    rows, cols = padded.shape
    return _Grid(
        padded=padded,
        rows=np.arange(half, rows - half, DOWNSAMPLING_FACTOR),
        cols=np.arange(half, cols - half, DOWNSAMPLING_FACTOR),
        kernels=[_scale_kernel(size) for size in sizes],
    )


def _row_high_freq(grid: _Grid, row: int) -> np.ndarray:
    """Return |DCT| high-frequency coefficients of every patch centred on one grid row."""
    parts: list[np.ndarray] = []
    for kernel in grid.kernels:
        half = int(kernel.size / 2)
        band = grid.padded[row - half: row + half + 1, :]
        patches = sliding_window_view(band, kernel.size, axis=1)[:, grid.cols - half, :]
        coef = np.abs(np.matmul(np.matmul(kernel.dct, patches.transpose(1, 0, 2)), np.transpose(kernel.dct)))
        parts.append(coef[:, kernel.high_freq_index[0], kernel.high_freq_index[1]])
    return np.hstack(parts)


def _layer_matrix(grid: _Grid) -> np.ndarray:
    """Return the per-point smallest high-frequency values, in argpartition order."""
    layers = 1 + sum(_scale_sizes())
    width = len(grid.cols)
    out = np.empty((len(grid.rows) * width, layers))
    for index, row in enumerate(grid.rows):
        high_freq = _row_high_freq(grid, int(row))
        order = np.argpartition(high_freq, layers, axis=1)[:, :layers]
        out[index * width: (index + 1) * width] = np.take_along_axis(high_freq, order, axis=1)
    return out


def _max_pooled_map(grid: _Grid) -> np.ndarray:
    """Normalise each layer by its maximum and max-pool across layers per point."""
    layers = _layer_matrix(grid)
    layers = layers / layers.max(axis=0)
    return layers.max(axis=1).reshape(len(grid.rows), len(grid.cols))


def detect_blur_map(gray: np.ndarray) -> np.ndarray:
    """Return blur_detector.detectBlur(gray) for downsampling 4 and 4 scales.

    gray: uint8 single-channel image. Output: float64 map at gray's size,
    normalised to max 1 (lower = more blurred).
    """
    detector = blur_detector.BlurDetector(
        downsampling_factor=DOWNSAMPLING_FACTOR, num_scales=NUM_SCALES, show_progress=False,
    )
    out_rows, out_cols = np.shape(gray)
    smoothed = cv2.GaussianBlur(gray, PRE_BLUR_KSIZE, sigmaX=PRE_BLUR_SIGMA, sigmaY=PRE_BLUR_SIGMA)
    t_max = _max_pooled_map(_build_grid(detector.computeImageGradientMagnitude(smoothed)))
    weighted = np.multiply(detector.entropyFilt(t_max), t_max)
    rows, cols = weighted.shape
    guide = cv2.GaussianBlur(
        cv2.resize(smoothed, (cols, rows)), PRE_BLUR_KSIZE, sigmaX=GUIDE_BLUR_SIGMA, sigmaY=GUIDE_BLUR_SIGMA,
    )
    final = cv2.resize(detector.RF(weighted, guide), (out_cols, out_rows))
    return final / np.max(final)
