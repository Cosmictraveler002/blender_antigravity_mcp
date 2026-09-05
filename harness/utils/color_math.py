"""
Consolidated Color Mathematics Utilities
=========================================
Single canonical implementation of sRGB ↔ CIE XYZ ↔ CIE L*a*b* conversions,
CIEDE2000 perceptual color difference, and related utilities.

Eliminates 4+ duplicate implementations previously scattered across:
  - verify_and_refine_bottle.py
  - test_accuracy.py
  - render_geometry_comparator.py
  - color_texture_analyzer.py
  - iterative_blender_loop.py
"""

import math
import numpy as np
from typing import Tuple

try:
    from skimage.color import deltaE_ciede2000 as _skimage_ciede2000
    HAS_SKIMAGE_DELTA_E = True
except ImportError:
    HAS_SKIMAGE_DELTA_E = False


# ─── sRGB ↔ Linear sRGB ──────────────────────────────────────────────────────

def srgb_to_linear(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Converts display sRGB [0,1] to linear sRGB for Blender node inputs."""
    def _to_lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (_to_lin(rgb[0]), _to_lin(rgb[1]), _to_lin(rgb[2]))


def linear_to_srgb(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Converts linear sRGB to display sRGB [0,1]."""
    def _to_srgb(c: float) -> float:
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return (_to_srgb(rgb[0]), _to_srgb(rgb[1]), _to_srgb(rgb[2]))


# ─── sRGB → CIE XYZ (D65) ────────────────────────────────────────────────────

# IEC 61966-2-1 sRGB to XYZ matrix (D65 illuminant)
_SRGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float64)


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """
    Convert normalized sRGB [0, 1] to CIE XYZ (D65 standard illuminant).

    Args:
        rgb: Array of shape (..., 3) with sRGB values in [0, 1].

    Returns:
        Array of same shape with CIE XYZ values.
    """
    a = 0.055
    linear = np.where(rgb > 0.04045, ((rgb + a) / (1.0 + a)) ** 2.4, rgb / 12.92)
    return np.dot(linear, _SRGB_TO_XYZ_MATRIX.T)


# ─── CIE XYZ → CIE L*a*b* (D65) ─────────────────────────────────────────────

# D65 standard illuminant tristimulus values
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_LAB_EPSILON = 216.0 / 24389.0  # ≈ 0.008856
_LAB_KAPPA = 24389.0 / 27.0     # ≈ 903.3


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """
    Convert CIE XYZ to CIE L*a*b* (D65 standard illuminant).

    Args:
        xyz: Array of shape (..., 3) with XYZ values.

    Returns:
        Array of same shape with L*a*b* values.
    """
    normalized = xyz / _D65_WHITE
    f = np.where(
        normalized > _LAB_EPSILON,
        normalized ** (1.0 / 3.0),
        (_LAB_KAPPA * normalized + 16.0) / 116.0
    )
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ─── Convenience: sRGB → Lab (single color or array) ─────────────────────────

def rgb_to_lab(rgb_float: Tuple[float, float, float]) -> np.ndarray:
    """
    Convert a single sRGB color (normalized [0,1]) to CIE L*a*b*.

    Args:
        rgb_float: Tuple of (R, G, B) in [0, 1].

    Returns:
        1D numpy array [L, a, b].
    """
    arr = np.array(rgb_float, dtype=np.float64)
    xyz = srgb_to_xyz(arr)
    return xyz_to_lab(xyz)


def rgb_array_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    """
    Convert an array of sRGB values to CIE L*a*b*.
    Accepts normalized float arrays or uint8 arrays.

    Args:
        rgb_array: Array of shape (..., 3). If dtype is uint8, auto-normalizes.

    Returns:
        Array of same shape with L*a*b* values.
    """
    if rgb_array.dtype == np.uint8:
        rgb_array = rgb_array.astype(np.float64) / 255.0
    elif rgb_array.dtype != np.float64:
        rgb_array = rgb_array.astype(np.float64)
    xyz = srgb_to_xyz(rgb_array)
    return xyz_to_lab(xyz)


# ─── CIEDE2000 Color Difference ──────────────────────────────────────────────

def compute_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Compute the CIEDE2000 perceptual color difference (dE2000) between two Lab colors.
    Uses scikit-image's implementation when available, falls back to CIE76 Euclidean distance.

    Args:
        lab1: 1D array [L, a, b] for color 1.
        lab2: 1D array [L, a, b] for color 2.

    Returns:
        Scalar dE2000 value. Values < 1.0 are imperceptible, > 5.0 are clearly different.
    """
    if HAS_SKIMAGE_DELTA_E:
        l1 = np.array([[[lab1[0], lab1[1], lab1[2]]]], dtype=np.float64)
        l2 = np.array([[[lab2[0], lab2[1], lab2[2]]]], dtype=np.float64)
        return float(_skimage_ciede2000(l1, l2)[0, 0])
    # Euclidean fallback (CIE76)
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


def compute_ciede2000_array(lab_image1: np.ndarray, lab_image2: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel CIEDE2000 between two Lab images.

    Args:
        lab_image1: Array of shape (H, W, 3) in L*a*b*.
        lab_image2: Array of shape (H, W, 3) in L*a*b*.

    Returns:
        Array of shape (H, W) with per-pixel dE2000 values.
    """
    if HAS_SKIMAGE_DELTA_E:
        return _skimage_ciede2000(
            lab_image1.astype(np.float64),
            lab_image2.astype(np.float64)
        )
    # Euclidean fallback
    return np.linalg.norm(lab_image1 - lab_image2, axis=-1)


# ─── Procrustes Shape Distance ───────────────────────────────────────────────

def compute_procrustes_distance(curve1: np.ndarray, curve2: np.ndarray) -> float:
    """
    Computes mean-squared shape discrepancy between two 1D radial contour arrays.
    Both curves are normalized to zero mean and unit variance before comparison.

    Args:
        curve1: 1D numpy array representing first radial profile.
        curve2: 1D numpy array representing second radial profile (same length).

    Returns:
        Scalar mean-squared distance. Lower is better; 0.0 = identical shapes.
    """
    c1 = (curve1 - np.mean(curve1)) / max(1e-5, np.std(curve1))
    c2 = (curve2 - np.mean(curve2)) / max(1e-5, np.std(curve2))
    return float(np.mean((c1 - c2) ** 2))


# ─── Fidelity Scoring ────────────────────────────────────────────────────────

def delta_e_to_fidelity(delta_e: float) -> float:
    """
    Maps a CIEDE2000 delta_e value to a 0-100 fidelity percentage.
    dE=0 → 100%, dE=5 → ~60%, dE=20 → ~0%.

    Args:
        delta_e: CIEDE2000 color difference value.

    Returns:
        Fidelity score in [0, 100].
    """
    return max(0.0, min(100.0, 100.0 * math.exp(-delta_e / 8.0)))


# ─── Display Utilities ───────────────────────────────────────────────────────

def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """Convert normalized RGB [0,1] to hex color string '#rrggbb'."""
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    """Convert hex color string '#rrggbb' to normalized RGB [0,1]."""
    hex_str = hex_str.lstrip('#')
    return (
        int(hex_str[0:2], 16) / 255.0,
        int(hex_str[2:4], 16) / 255.0,
        int(hex_str[4:6], 16) / 255.0
    )
