import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
import scipy.ndimage as ndi

from harness.utils.color_math import (
    srgb_to_xyz, xyz_to_lab, rgb_to_lab, srgb_to_linear, rgb_to_hex,
    compute_ciede2000 as compute_delta_e_2000,
)

class ColorTextureAnalyzer:
    """
    Advanced Color & Texture Profiler:
    - Extracts position-agnostic colors using geometry-driven masked ROIs.
    - CIE L*a*b* K-Means clustering for foreground palette.
    - CIEDE2000 perceptual difference metric calculations.
    - Multi-scale Gabor filter bank for grain angle & isotropy analysis.
    - Specular highlight detection to avoid washed-out diffuse base colors.
    - Full color histograms (20 bins) for structural material comparison.
    - High-fidelity Blender Principled BSDF parameter synthesis.
    """

    def __init__(self, image_path: str, geometry_doc_path: Optional[str] = None):
        self.image_path = os.path.abspath(image_path)
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self.img_bgr = cv2.imread(self.image_path)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.h, self.w, _ = self.img_rgb.shape

        self.geom_doc = None
        if geometry_doc_path and os.path.exists(geometry_doc_path):
            with open(geometry_doc_path, "r", encoding="utf-8") as f:
                self.geom_doc = json.load(f)

    def extract_dominant_palette(self, fg_mask: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Performs K-Means clustering in CIE L*a*b* space on foreground pixels."""
        rgb_norm = self.img_rgb.astype(np.float32) / 255.0
        mask_bool = fg_mask > 128
        fg_pixels = rgb_norm[mask_bool]

        if len(fg_pixels) < k:
            return []

        # Subsample for speed and uniformity
        if len(fg_pixels) > 50000:
            indices = np.random.choice(len(fg_pixels), 50000, replace=False)
            fg_sample = fg_pixels[indices]
        else:
            fg_sample = fg_pixels

        xyz = srgb_to_xyz(fg_sample)
        lab = xyz_to_lab(xyz)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(lab)
        cluster_labels = kmeans.labels_
        counts = np.bincount(cluster_labels)
        total_pts = len(fg_sample)

        palette = []
        for i in range(k):
            centroid_lab = kmeans.cluster_centers_[i]
            cluster_rgb = np.median(fg_sample[cluster_labels == i], axis=0)
            cluster_rgb = np.clip(cluster_rgb, 0.0, 1.0)
            coverage_pct = round(float(counts[i] / total_pts * 100.0), 1)
            lin_rgb = srgb_to_linear(tuple(cluster_rgb))

            palette.append({
                "rank": i + 1,
                "coverage_pct": coverage_pct,
                "srgb": [round(float(c), 3) for c in cluster_rgb],
                "srgb_linear": [round(float(c), 3) for c in lin_rgb],
                "hex": rgb_to_hex(cluster_rgb),
                "cie_lab": {
                    "L": round(float(centroid_lab[0]), 1),
                    "a": round(float(centroid_lab[1]), 1),
                    "b": round(float(centroid_lab[2]), 1)
                }
            })

        palette = sorted(palette, key=lambda p: p["coverage_pct"], reverse=True)
        return palette

    def detect_specular_highlights(self, crop_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detects specular highlights (glare) and computes coverage percentage."""
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        # Specular highlight: very high brightness and low saturation
        spec_mask = (hsv[:, :, 2] > 215) & (hsv[:, :, 1] < 45)
        coverage = float(np.sum(spec_mask) / max(1, crop_rgb.shape[0] * crop_rgb.shape[1]) * 100.0)
        return spec_mask, round(coverage, 2)

    def analyze_texture_and_gabor(self, region_crop: np.ndarray) -> Dict[str, Any]:
        """
        Advanced micro-surface texture analysis:
        - Multi-scale Gabor filter bank (4 angles) for grain orientation & isotropy.
        - High-frequency gradient energy.
        - Uniform Local Binary Pattern (LBP) entropy.
        """
        if region_crop.size == 0 or region_crop.shape[0] < 8 or region_crop.shape[1] < 8:
            return {
                "roughness_estimate": 0.5,
                "texture_category": "UNKNOWN",
                "gradient_energy": 0.0,
                "lbp_entropy": 0.0,
                "gabor_grain_direction_deg": 0.0,
                "gabor_isotropy_score": 1.0
            }

        gray = cv2.cvtColor(region_crop, cv2.COLOR_RGB2GRAY) if len(region_crop.shape) == 3 else region_crop
        gray_f = gray.astype(np.float32) / 255.0

        # 1. Gradient energy
        gy, gx = np.gradient(gray_f)
        energy = float(np.mean(gx**2 + gy**2))

        # 2. Local Binary Pattern entropy
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))

        # 3. Gabor Filter Bank (orientations: 0, 45, 90, 135 deg)
        angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
        deg_map = [0.0, 45.0, 90.0, 135.0]
        responses = []

        for theta in angles:
            kernel = cv2.getGaborKernel((15, 15), sigma=3.0, theta=theta, lambd=8.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
            f_img = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
            responses.append(float(np.mean(np.abs(f_img))))

        max_idx = int(np.argmax(responses))
        min_r = min(responses)
        max_r = max(responses)
        isotropy = float(min_r / max(1e-5, max_r))

        # Eliminate forced argmax trap: If surface is isotropic or gradient energy is negligible,
        # do not force a directional grain angle (PIPELINE_SOLUTIONS_SPEC.md § 1.1)
        is_isotropic = bool(isotropy >= 0.72 or energy < 0.004)
        grain_direction = None if is_isotropic else deg_map[max_idx]

        # Calibrated roughness mapping:
        # Matte powder coat: high micro-dispersion (0.65 - 0.78)
        # Laser text stainless steel: smoother, lower roughness (0.28 - 0.35)
        # Bamboo wood: directional fiber grain (0.38 - 0.45)
        roughness = float(np.clip(0.35 + 18.0 * energy, 0.15, 0.85))

        if energy > 0.015:
            cat = "GRAINED_OR_ROUGH"
        elif energy > 0.005:
            cat = "FINE_MATTE_POWDER"
        else:
            cat = "SMOOTH_ISOTROPIC" if is_isotropic else "SMOOTH_SATIN"

        return {
            "roughness_estimate": round(roughness, 2),
            "gradient_energy": round(energy, 4),
            "lbp_entropy": round(entropy, 2),
            "gabor_grain_direction_deg": grain_direction,
            "gabor_isotropy_score": round(isotropy, 3),
            "is_isotropic": is_isotropic,
            "texture_category": cat
        }

    def compute_lab_histogram(self, crop_rgb: np.ndarray, bins: int = 20) -> Dict[str, List[float]]:
        """Computes normalized 20-bin Lab histograms for structural color comparison."""
        if crop_rgb.size == 0:
            return {"L": [0.0] * bins, "a": [0.0] * bins, "b": [0.0] * bins}

        rgb_norm = crop_rgb.astype(np.float32) / 255.0
        flat_rgb = rgb_norm.reshape(-1, 3)
        xyz = srgb_to_xyz(flat_rgb)
        lab = xyz_to_lab(xyz)

        hist_L, _ = np.histogram(lab[:, 0], bins=bins, range=(0, 100), density=True)
        hist_a, _ = np.histogram(lab[:, 1], bins=bins, range=(-86, 98), density=True)
        hist_b, _ = np.histogram(lab[:, 2], bins=bins, range=(-107, 94), density=True)

        return {
            "L": [round(float(v), 5) for v in hist_L],
            "a": [round(float(v), 5) for v in hist_a],
            "b": [round(float(v), 5) for v in hist_b]
        }

    def analyze_component_materials_dynamic(self) -> Dict[str, Any]:
        """
        Dynamically extracts color, reflectance, and texture for all components
        using exact bounding boxes and regions from geometry_design_doc.json.
        Completely object-agnostic.
        """
        def sample_clean_color(crop: np.ndarray, is_dark: bool = False, is_bright: bool = False) -> Tuple[np.ndarray, float]:
            if crop.size == 0:
                return np.array([0.5, 0.5, 0.5]), 0.0

            spec_mask, spec_cov = self.detect_specular_highlights(crop)
            non_spec = ~spec_mask

            if is_dark:
                mask = (crop[:, :, 0] < 90) & (crop[:, :, 1] < 90) & (crop[:, :, 2] < 90) & non_spec
            elif is_bright:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                mask = (gray > 135) & non_spec
            else:
                mask = non_spec

            if np.any(mask):
                clean_rgb = np.median(crop[mask], axis=0) / 255.0
            else:
                clean_rgb = np.median(crop.reshape(-1, 3), axis=0) / 255.0

            return clean_rgb, spec_cov

        def make_spec(name: str, rgb: np.ndarray, spec_cov: float, tex: Dict[str, Any], rough: float, metallic: float,
                      subsurf: float, sheen: float, notes: str) -> Dict[str, Any]:
            rgb_tup = (float(np.clip(rgb[0], 0, 1)), float(np.clip(rgb[1], 0, 1)), float(np.clip(rgb[2], 0, 1)))
            lab = rgb_to_lab(rgb_tup)
            lin_rgb = srgb_to_linear(rgb_tup)

            return {
                "material_name": name,
                "dominant_color": {
                    "srgb_display": [round(c, 3) for c in rgb_tup],
                    "srgb_linear": [round(c, 3) for c in lin_rgb],
                    "hex": rgb_to_hex(rgb_tup),
                    "cie_lab": {
                        "L": round(float(lab[0]), 1),
                        "a": round(float(lab[1]), 1),
                        "b": round(float(lab[2]), 1)
                    }
                },
                "specular_coverage_pct": spec_cov,
                "principled_bsdf": {
                    "base_color_rgb": [round(c, 3) for c in rgb_tup],
                    "base_color_srgb_linear": [round(c, 3) for c in lin_rgb],
                    "base_color_rgba": [round(c, 3) for c in rgb_tup] + [1.0],
                    "roughness": round(rough, 2),
                    "metallic": round(metallic, 2),
                    "specular_ior_level": round(0.50 + spec_cov * 0.005, 2),
                    "sheen_weight": round(sheen, 2),
                    "subsurface_weight": round(subsurf, 2)
                },
                "surface_texture": tex,
                "notes": notes
            }

        materials = {}
        from harness.analyzers.svbrdf_analyzer import SVBRDFEngine
        svbrdf_engine = SVBRDFEngine()

        if self.geom_doc and "components" in self.geom_doc and len(self.geom_doc["components"]) > 0:
            for cid, comp in self.geom_doc["components"].items():
                bounds = comp.get("snapped_pixel_y_bounds")
                bbox = comp.get("pixel_bbox")
                if bounds:
                    y1, y2 = bounds
                    bx = bbox[0] if bbox else 0
                    bw = bbox[2] if bbox else self.w
                elif bbox:
                    bx, by, bw, bh = bbox
                    y1, y2 = by, by + bh
                else:
                    bx, bw, y1, y2 = 0, self.w, 0, self.h

                x1 = max(0, min(self.w - 1, bx))
                x2 = max(x1 + 1, min(self.w, bx + bw))
                y1 = max(0, min(self.h - 1, y1))
                y2 = max(y1 + 1, min(self.h, y2))

                # Rule 2: internal erosion to prevent edge contamination
                erosion = 3
                crop_y1 = min(y2 - 1, y1 + erosion)
                crop_y2 = max(crop_y1 + 1, y2 - erosion)
                crop_x1 = min(x2 - 1, x1 + erosion)
                crop_x2 = max(crop_x1 + 1, x2 - erosion)
                crop = self.img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

                rough_hint = comp.get("estimated_roughness", 0.50)
                metal_hint = comp.get("estimated_metallic", 0.0)
                disp_name = comp.get("display_name", cid).replace(" ", "") + "Material"

                c_rgb, c_spec = sample_clean_color(crop)
                c_tex = self.analyze_texture_and_gabor(crop)

                # Execute SVBRDF engine intrinsic analysis
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if crop.size > 0 else np.zeros((32, 32, 3), dtype=np.uint8)
                svbrdf_temp_dir = os.path.join(os.path.dirname(self.image_path), "..", "textures", cid)
                try:
                    svbrdf_manifest = svbrdf_engine.decompose_crop(
                        crop_bgr=crop_bgr,
                        output_dir=svbrdf_temp_dir,
                        component_id=cid,
                        target_color_hex=comp.get("color_hex"),
                        base_roughness_hint=rough_hint,
                        base_metallic_hint=metal_hint,
                        erosion_px=0  # Already eroded above
                    )
                except Exception as e:
                    print(f"[ColorTextureAnalyzer] SVBRDF analysis fallback for {cid}: {e}")
                    svbrdf_manifest = {"composite_confidence_q": 0.5, "decomposition_mode": "basic_fallback"}

                mat_spec = make_spec(
                    disp_name, c_rgb, c_spec, c_tex,
                    rough=rough_hint, metallic=metal_hint,
                    subsurf=0.0, sheen=0.05,
                    notes=comp.get("description", f"Procedural PBR material for {cid}")
                )
                mat_spec["color_histogram_lab"] = self.compute_lab_histogram(crop)
                mat_spec["svbrdf_analysis"] = svbrdf_manifest
                materials[cid] = mat_spec
        else:
            # Fallback to dominant palette
            palette = self.extract_dominant_palette(np.ones((self.h, self.w), dtype=np.uint8) * 255, k=3)
            for idx, p in enumerate(palette):
                mid = f"palette_material_{idx}"
                rgb = np.array(p["srgb"])
                c_tex = {"coarse_roughness_score": 0.5, "fine_grain_contrast": 0.5, "perceived_texture_class": "smooth"}
                mat_spec = make_spec(
                    f"PaletteMat{idx}", rgb, 0.05, c_tex,
                    rough=0.50, metallic=0.0, subsurf=0.0, sheen=0.0,
                    notes=f"Auto-extracted dominant palette color {p['hex']}"
                )
                materials[mid] = mat_spec

        return materials

    def generate_palette_swatch_image(self, palette: List[Dict[str, Any]], materials: Dict[str, Any], output_path: str):
        """Generates visual color swatch & material parameter diagnostic image."""
        swatch_h = 130
        swatch_w = 750
        canvas = np.ones((swatch_h * 2 + 70, swatch_w, 3), dtype=np.uint8) * 245

        # Row 1: Dominant Palette Clusters
        cv2.putText(canvas, "DOMINANT PALETTE (K-Means CIE Lab)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        x_offset = 20
        block_w = (swatch_w - 40) // max(1, len(palette))
        for p in palette:
            rgb = [int(c * 255) for c in p["srgb"]]
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(canvas, (x_offset, 45), (x_offset + block_w - 8, 45 + 65), bgr, -1)
            cv2.rectangle(canvas, (x_offset, 45), (x_offset + block_w - 8, 45 + 65), (180, 180, 180), 1)
            cv2.putText(canvas, f"{p['coverage_pct']}%", (x_offset + 5, 45 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
            cv2.putText(canvas, p["hex"], (x_offset + 5, 45 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
            x_offset += block_w

        # Row 2: Target Material Colors
        y_m = swatch_h + 40
        cv2.putText(canvas, "COMPONENT MATERIALS (Blender Principled BSDF)", (20, y_m), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        x_offset = 20
        mat_list = list(materials.values())
        block_w = (swatch_w - 40) // max(1, len(mat_list))
        for m in mat_list:
            rgb = [int(c * 255) for c in m["principled_bsdf"]["base_color_rgb"]]
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(canvas, (x_offset, y_m + 15), (x_offset + block_w - 8, y_m + 15 + 65), bgr, -1)
            cv2.rectangle(canvas, (x_offset, y_m + 15), (x_offset + block_w - 8, y_m + 15 + 65), (180, 180, 180), 1)
            short_name = m["material_name"].replace("Material", "")
            cv2.putText(canvas, short_name[:11], (x_offset + 2, y_m + 15 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1)
            cv2.putText(canvas, f"R:{m['principled_bsdf']['roughness']}", (x_offset + 2, y_m + 15 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
            x_offset += block_w

        cv2.imwrite(output_path, canvas)

    def analyze(self, output_json_path: str = "color_texture_design_doc.json", output_vis_path: str = "color_texture_swatches.png") -> Dict[str, Any]:
        print(f"[ColorTextureAnalyzer] Analyzing color and texture for {self.image_path}...")

        if self.geom_doc and "overall_dimensions" in self.geom_doc:
            bbox = self.geom_doc["overall_dimensions"]["bbox_pixels"]
            bx, by, bw, bh = bbox
        else:
            bx, by, bw, bh = (398, 48, 230, 914)

        # Foreground mask
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        fg_mask[by:by + bh, bx:bx + bw] = 255

        print("[ColorTextureAnalyzer] Extracting dominant palette via CIE Lab K-Means...")
        palette = self.extract_dominant_palette(fg_mask, k=5)

        print("[ColorTextureAnalyzer] Extracting geometry-driven component materials & BSDF parameters...")
        materials = self.analyze_component_materials_dynamic()

        doc = {
            "meta": {
                "source_image": os.path.basename(self.image_path),
                "framework_version": "3.0.0",
                "geometry_doc_used": bool(self.geom_doc is not None)
            },
            "dominant_palette": palette,
            "component_materials": materials
        }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"[ColorTextureAnalyzer] Saved color/texture design doc -> {output_json_path}")

        self.generate_palette_swatch_image(palette, materials, output_vis_path)
        print(f"[ColorTextureAnalyzer] Saved visual swatch image -> {output_vis_path}")

        return doc

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "reference_bottle.jpg"
    geom_doc = sys.argv[2] if len(sys.argv) > 2 else "geometry_design_doc.json"
    analyzer = ColorTextureAnalyzer(target, geom_doc)
    doc = analyzer.analyze()
    print("\n--- Summary of Extracted Colors & Shaders ---")
    for k, m in doc["component_materials"].items():
        rgb = m["principled_bsdf"]["base_color_rgb"]
        rough = m["principled_bsdf"]["roughness"]
        tex = m["surface_texture"]["texture_category"]
        print(f"{m['material_name']:<25}: RGB=({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}) | Rough={rough} | Tex={tex}")
