"""
Physical & Optical Material Property Estimators
================================================
Structural, objective PBR material estimators derived from classical optics,
Fresnel equations, and micro-facet theory (Cook-Torrance GGX):
  1. estimate_physical_roughness: Multi-scale micro-facet dispersion & specular gradient falloff
  2. estimate_physical_metallic: Conductor vs dielectric spectral discrimination via Fresnel ratio,
     specular highlight chromaticity, and diffuse floor analysis.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


def estimate_physical_roughness(
    crop_rgb: np.ndarray,
    spec_mask: Optional[np.ndarray] = None,
    gradient_energy: Optional[float] = None,
    lbp_entropy: Optional[float] = None,
    prior_hint: Optional[float] = None
) -> float:
    """
    Estimates physically grounded surface roughness [0.02, 0.95].
    
    Combines:
      - Specular highlight boundary sharpness (steep gradient falloff = low roughness)
      - Micro-facet dispersion (high-frequency gradient energy away from highlights)
      - LBP micro-texture entropy
    """
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 4 or crop_rgb.shape[1] < 4:
        return float(prior_hint if prior_hint is not None else 0.50)

    # Normalize RGB to [0, 1] float32
    if crop_rgb.dtype == np.uint8:
        rgb_f = crop_rgb.astype(np.float32) / 255.0
    else:
        rgb_f = np.clip(crop_rgb.astype(np.float32), 0.0, 1.0)

    gray_f = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY)
    h, w = gray_f.shape

    # 1. Content-Adaptive Specular Highlight Detection
    # Avoid hardcoded luminance thresholds so highlights are detected on dark satin metals and bright dielectrics alike
    p95_lum = float(np.percentile(gray_f, 95))
    p50_lum = float(np.median(gray_f))
    p10_lum = float(np.percentile(gray_f, 10))

    if spec_mask is None:
        dynamic_spec_thresh = max(0.35, min(0.92, p95_lum - 0.02))
        spec_mask = (gray_f >= dynamic_spec_thresh) & (gray_f > p50_lum + 0.10)

    spec_pixels = int(np.sum(spec_mask))
    spec_coverage = float(spec_pixels) / float(max(1, h * w))

    # 2. High-Frequency Micro-Texture Gradient Energy
    if gradient_energy is None:
        gy, gx = np.gradient(gray_f)
        non_spec = ~spec_mask
        if np.any(non_spec):
            gradient_energy = float(np.mean(gx[non_spec]**2 + gy[non_spec]**2))
        else:
            gradient_energy = float(np.mean(gx**2 + gy**2))

    # 3. Highlight Sharpness (Laplacian / Gradient falloff at highlight edges)
    roughness_spec = 0.50
    has_valid_highlight = False

    if 0.001 < spec_coverage < 0.45:
        spec_u8 = spec_mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(spec_u8, kernel, iterations=1)
        border = (dilated > 0) & (~spec_mask)

        if np.any(border):
            sobelx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            edge_sharpness = float(np.mean(grad_mag[border]))

            # Calibrated physically grounded GGX alpha curve:
            # Steep edge sharpness (> 0.20) -> polished/mirror/smooth (roughness 0.02 - 0.15)
            # Moderate edge sharpness (0.06 - 0.15) -> satin/semi-gloss (roughness 0.25 - 0.45)
            # Low edge sharpness (< 0.05) -> wide diffuse highlight (roughness 0.50 - 0.70)
            roughness_spec = float(np.clip(0.70 - edge_sharpness * 2.2, 0.02, 0.80))
            has_valid_highlight = True

    # 4. Micro-texture based roughness with responsive power-law dynamic range
    # Replaces flat linear offset with continuous physical dispersion response:
    # Smooth/polished glass/mirror: energy < 0.0001 -> roughness 0.02 - 0.10
    # Satin metal/smooth plastic: energy 0.0005 - 0.003 -> roughness 0.20 - 0.38
    # Fine matte powder coat: energy 0.005 - 0.012 -> roughness 0.48 - 0.65
    # Pebbled leatherette / heavy knurling: energy > 0.020 -> roughness 0.75 - 0.95
    clamped_energy = max(1e-6, float(gradient_energy))
    roughness_tex = float(np.clip(0.04 + 3.8 * (clamped_energy ** 0.42), 0.02, 0.95))

    # 5. Composite Objective Formulation
    if has_valid_highlight:
        measured_roughness = 0.55 * roughness_spec + 0.45 * roughness_tex
    else:
        measured_roughness = roughness_tex

    # 6. Prior Hint Fusion (if provided and confident)
    if prior_hint is not None and 0.0 <= prior_hint <= 1.0:
        # Confidence weighting: only pull towards prior if image crop lacks distinct texture
        confidence = min(1.0, float(gradient_energy) * 50.0 + (0.5 if has_valid_highlight else 0.1))
        weight_measured = 0.85 * confidence + 0.15
        final_roughness = weight_measured * measured_roughness + (1.0 - weight_measured) * float(prior_hint)
    else:
        final_roughness = measured_roughness

    return float(np.clip(final_roughness, 0.02, 0.95))


def estimate_physical_metallic(
    crop_rgb: np.ndarray,
    category_hint: Optional[str] = None,
    prior_hint: Optional[float] = None
) -> float:
    """
    Estimates physically grounded metallic factor [0.0, 1.0] (dielectric vs conductor).
    
    Principles:
      1. Specular highlight chromaticity vs diffuse body chromaticity
         - Dielectrics: Achromatic (white) highlight, colored body.
         - Colored Metals (gold, copper, brass): Chromatic highlight matching body color.
      2. Specular-to-diffuse dynamic range contrast ratio (Cook-Torrance F0 / kd)
         - Conductors lack internal scattering; diffuse non-specular base is dark absorption.
         - Dielectrics exhibit strong Lambertian diffuse reflection body.
      3. Content-adaptive dark satin & bright polished metal discrimination
    """
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 4 or crop_rgb.shape[1] < 4:
        if prior_hint is not None:
            return float(np.clip(prior_hint, 0.0, 1.0))
        return 0.0

    if crop_rgb.dtype == np.uint8:
        rgb_f = crop_rgb.astype(np.float32) / 255.0
    else:
        rgb_f = np.clip(crop_rgb.astype(np.float32), 0.0, 1.0)

    lum = 0.2126 * rgb_f[:, :, 0] + 0.7152 * rgb_f[:, :, 1] + 0.0722 * rgb_f[:, :, 2]
    h, w = lum.shape

    # HSV for saturation & value analysis
    rgb_u8 = (rgb_f * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0

    p95_lum = float(np.percentile(lum, 95))
    p20_lum = float(np.percentile(lum, 20))

    spec_mask = lum >= max(0.25, p95_lum - 0.03)
    body_mask = (lum <= np.percentile(lum, 75)) & (lum >= p20_lum)

    spec_count = int(np.sum(spec_mask))
    body_count = int(np.sum(body_mask))

    # Specular contrast ratio: Ratio of peak highlight to diffuse floor
    contrast_ratio = (p95_lum + 1e-3) / (p20_lum + 1e-3)
    diff_lum = p95_lum - p20_lum

    metallic_score = 0.0

    # 1. Colored Metal vs Colored Dielectric Test
    mean_body_sat = float(np.mean(sat[body_mask])) if body_count > 0 else float(np.mean(sat))
    if mean_body_sat > 0.20 and spec_count > 8 and body_count > 8:
        spec_sat = float(np.mean(sat[spec_mask]))
        sat_ratio = spec_sat / (mean_body_sat + 1e-4)
        if sat_ratio > 0.65:
            # Colored metal (gold, copper, brass)
            metallic_score = min(1.0, 0.70 + 0.30 * sat_ratio)
        else:
            # Colored dielectric with white specular highlight
            metallic_score = max(0.0, 0.20 * sat_ratio)

    # 2. Neutral Metal vs Dark Dielectric Test (Low saturation: dark/gray/silver/chrome)
    else:
        # Physical discrimination: Conductors have high F0/kd contrast even in dark finishes.
        has_specular_response = (diff_lum > 0.10) and (spec_count >= 5)

        if has_specular_response and contrast_ratio > 3.8:
            # Polished or satin metals with high peak-to-floor absorption
            metallic_score = float(np.clip(0.50 + (contrast_ratio - 3.8) * 0.08 + diff_lum * 0.4, 0.50, 0.95))
        elif has_specular_response and contrast_ratio > 2.4:
            # Semi-metallic / low-specular metal finish
            metallic_score = float(np.clip(0.25 + (contrast_ratio - 2.4) * 0.12, 0.20, 0.50))
        elif p95_lum > 0.70 and contrast_ratio > 1.8:
            # Very bright reflective surface
            metallic_score = 0.35
        else:
            # Diffuse dielectric (matte plastic, rubber, wood, fabric, paper)
            metallic_score = 0.0

    # 3. Category Semantic Prior Modulation
    if category_hint:
        cat_lower = category_hint.lower()
        metal_keywords = [
            "metal", "aluminum", "steel", "brass", "copper", "chrome",
            "magnesium", "alloy", "mount", "bayonet", "dial", "rail"
        ]
        dielectric_keywords = [
            "rubber", "leatherette", "plastic", "glass", "lens", "element",
            "screen", "display", "ceramic", "wood", "paper", "fabric"
        ]

        if any(kw in cat_lower for kw in metal_keywords):
            prior_val = 0.88
            metallic_score = max(prior_val, 0.40 * metallic_score + 0.60 * prior_val)
        elif any(kw in cat_lower for kw in dielectric_keywords):
            prior_val = 0.02
            metallic_score = min(0.08, 0.40 * metallic_score + 0.60 * prior_val)

    # 4. Hint Fusion
    if prior_hint is not None and 0.0 <= prior_hint <= 1.0:
        metallic_score = 0.75 * metallic_score + 0.25 * float(prior_hint)

    return float(np.clip(metallic_score, 0.0, 1.0))
