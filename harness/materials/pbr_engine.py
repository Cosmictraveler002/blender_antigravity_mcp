"""
PBR Texture Retrieval & Download Engine
========================================
Implements the 6-step PBR asset retrieval and procedural synthesis pipeline:
  1. Reads semantic material descriptors (from Gemini Vision / ColorTextureAnalyzer)
  2. Searches open PBR repositories (PolyHaven & ambientCG)
  3. Ranks candidates by keyword, category, and tag similarity
  4. Downloads complete PBR map sets (albedo/diffuse, roughness, normal_gl, AO)
  5. Fallback: synthesizes procedural PBR maps if match score < threshold or offline
  6. Emits structured material manifest for Blender node graph construction
"""

import os
import sys
import json
import shutil
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import time
import concurrent.futures


USER_AGENT = "BlenderMCP/4.0 (Windows; Python-Harness)"


class PolyHavenClient:
    """Client for the public, free, CC0 PolyHaven Assets API."""

    BASE_API = "https://api.polyhaven.com"

    def __init__(self):
        self._assets_cache: Optional[Dict[str, Any]] = None

    def get_all_textures(self) -> Dict[str, Any]:
        """Fetch and cache list of all available textures."""
        if self._assets_cache is not None:
            return self._assets_cache

        cache_dir = os.path.join(os.path.expanduser("~"), ".blender_mcp", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "catalogue_polyhaven.json")
        
        # Check 24-hour TTL
        if os.path.isfile(cache_file):
            if time.time() - os.path.getmtime(cache_file) < 86400:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        self._assets_cache = json.load(f)
                    return self._assets_cache
                except Exception as e:
                    print(f"[PolyHaven] Cache read failed: {e}")

        url = f"{self.BASE_API}/assets?t=textures"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read().decode("utf-8")
                self._assets_cache = json.loads(data)
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(data)
        except Exception as e:
            print(f"[PolyHaven] WARNING: Could not fetch texture catalogue: {e}")
            self._assets_cache = {}

        return self._assets_cache

    def search(self, keywords: List[str], category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Rank textures based on keyword and category overlap.

        Args:
            keywords: List of search terms (e.g. ['bamboo', 'wood', 'planks']).
            category: Optional category filter (e.g. 'wood', 'metal').
            limit: Maximum candidates to return.

        Returns:
            List of ranked candidates with scores.
        """
        catalogue = self.get_all_textures()
        if not catalogue:
            return []

        search_tokens = set()
        for kw in keywords:
            for token in kw.lower().replace("_", " ").replace("-", " ").split():
                if len(token) > 2:
                    search_tokens.add(token)

        candidates = []
        for asset_id, meta in catalogue.items():
            score = 0.0
            asset_name = asset_id.lower()
            categories = [c.lower() for c in meta.get("categories", [])]
            tags = [t.lower() for t in meta.get("tags", [])]

            # Category match bonus
            if category and (category.lower() in categories or any(category.lower() in c for c in categories)):
                score += 3.0

            # Direct token match
            primary_token = keywords[0].lower().strip() if keywords else ""
            for token in search_tokens:
                if token in asset_name:
                    score += 10.0
                    if token == primary_token:
                        score += 15.0  # Massive bonus for matching primary specific keyword (e.g. bamboo)
                for cat in categories:
                    if token in cat:
                        score += 2.0
                for tag in tags:
                    if token == tag:
                        score += 3.0
                    elif token in tag:
                        score += 1.0

            if score > 0:
                candidates.append({
                    "source": "polyhaven",
                    "asset_id": asset_id,
                    "score": score,
                    "categories": categories,
                    "tags": tags[:6],
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:limit]

    def get_map_urls(self, asset_id: str, resolution: str = "1k") -> Dict[str, str]:
        """
        Retrieve direct download URLs for PBR texture maps.

        Returns dict with keys: 'diffuse', 'roughness', 'normal', 'ao', 'displacement'.
        """
        url = f"{self.BASE_API}/files/{asset_id}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[PolyHaven] WARNING: Could not fetch files for {asset_id}: {e}")
            return {}

        map_urls = {}

        # 1. Diffuse / Base Color
        if "Diffuse" in data and resolution in data["Diffuse"]:
            res_data = data["Diffuse"][resolution]
            fmt = "jpg" if "jpg" in res_data else ("png" if "png" in res_data else list(res_data.keys())[0])
            map_urls["diffuse"] = res_data[fmt]["url"]

        # 2. Roughness
        if "Rough" in data and resolution in data["Rough"]:
            res_data = data["Rough"][resolution]
            fmt = "jpg" if "jpg" in res_data else ("png" if "png" in res_data else list(res_data.keys())[0])
            map_urls["roughness"] = res_data[fmt]["url"]

        # 3. Normal (prefer OpenGL format nor_gl)
        normal_key = "nor_gl" if "nor_gl" in data else ("nor_dx" if "nor_dx" in data else None)
        if normal_key and resolution in data[normal_key]:
            res_data = data[normal_key][resolution]
            fmt = "jpg" if "jpg" in res_data else ("png" if "png" in res_data else list(res_data.keys())[0])
            map_urls["normal"] = res_data[fmt]["url"]

        # 4. Ambient Occlusion
        if "AO" in data and resolution in data["AO"]:
            res_data = data["AO"][resolution]
            fmt = "jpg" if "jpg" in res_data else ("png" if "png" in res_data else list(res_data.keys())[0])
            map_urls["ao"] = res_data[fmt]["url"]

        # 5. Displacement
        if "Displacement" in data and resolution in data["Displacement"]:
            res_data = data["Displacement"][resolution]
            fmt = "jpg" if "jpg" in res_data else ("png" if "png" in res_data else list(res_data.keys())[0])
            map_urls["displacement"] = res_data[fmt]["url"]

        return map_urls


class AmbientCGClient:
    """Client for the public, free, CC0 ambientCG PBR API."""

    BASE_API = "https://ambientcg.com/api/v2"

    def search(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Search ambientCG materials by query words."""
        query = "+".join(keywords[:3])
        url = f"{self.BASE_API}/full_json?type=Material&q={urllib.parse.quote(query)}&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[ambientCG] WARNING: Search failed: {e}")
            return []

        found = data.get("foundAssets", [])
        candidates = []
        for a in found:
            candidates.append({
                "source": "ambientcg",
                "asset_id": a.get("assetId"),
                "score": float(a.get("quality", 10.0)),
                "categories": [a.get("dataType", "Material")],
                "tags": a.get("tags", []),
            })

        return candidates


class PBRMaterialEngine:
    """
    Main orchestration engine for resolving, downloading, caching,
    and synthesizing PBR texture maps for any 3D reconstruction component.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.abspath(os.path.join(os.getcwd(), "assets", "textures_cache"))
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.polyhaven = PolyHavenClient(cache_dir=self.cache_dir)
        self.ambientcg = AmbientCGClient()
        self._downloaded_assets: Dict[str, Dict[str, str]] = {}
        self._session_asset_cache: Dict[str, Dict[str, str]] = {}

    def resolve_material(
        self,
        component_id: str,
        keywords: List[str],
        category: str,
        target_dir: str,
        target_color_hex: Optional[str] = None,
        estimated_roughness: float = 0.5,
        estimated_metallic: float = 0.0,
        resolution: str = "1k",
        min_score: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Find and download the best matching PBR texture set for a component.

        Args:
            component_id: Unique identifier for component (e.g., 'bamboo_cap').
            keywords: Descriptive keywords from Gemini vision analysis.
            category: High-level material category (e.g. 'wood', 'metal').
            target_dir: Directory to save resolved maps (projects/<p>/outputs/textures/<comp>).
            target_color_hex: Target hex color for procedural tint/fallback.
            estimated_roughness: Target roughness estimate [0, 1].
            estimated_metallic: Target metallic factor [0, 1].
            resolution: '1k' or '2k'.
            min_score: Minimum match score threshold to accept a downloaded texture.

        Returns:
            Dict describing resolved maps and material configuration.
        """
        os.makedirs(target_dir, exist_ok=True)
        print(f"[PBREngine] Resolving material for '{component_id}' (Category: {category}, Keywords: {keywords})")

        # 1. Search PolyHaven (primary)
        ph_candidates = self.polyhaven.search(keywords=keywords, category=category, limit=3)

        best_match = None
        if ph_candidates and ph_candidates[0]["score"] >= min_score:
            best_match = ph_candidates[0]

        # 2. Try downloading if we have a qualified match
        if best_match:
            asset_id = best_match["asset_id"]
            downloaded = None

            if asset_id in self._downloaded_assets:
                print(f"[PBREngine] Reusing cached asset '{asset_id}' from current run.")
                downloaded = {}
                for map_type, src_path in self._downloaded_assets[asset_id].items():
                    dest_file = os.path.join(target_dir, os.path.basename(src_path))
                    shutil.copyfile(src_path, dest_file)
                    downloaded[map_type] = dest_file
            else:
                print(f"[PBREngine] Selected '{asset_id}' from PolyHaven (Score: {best_match['score']:.1f})")
                downloaded = self._download_polyhaven_asset(asset_id, target_dir, resolution=resolution)
                if downloaded:
                    self._downloaded_assets[asset_id] = downloaded.copy()

            if downloaded:
                downloaded = self._validate_downloaded_maps(downloaded, target_dir, category, target_color_hex)
                return {
                    "component_id": component_id,
                    "status": "pbr_downloaded",
                    "source": "polyhaven",
                    "asset_id": asset_id,
                    "maps": downloaded,
                    "target_color_hex": target_color_hex,
                    "estimated_roughness": estimated_roughness,
                    "estimated_metallic": estimated_metallic,
                    "is_procedural_fallback": False,
                }

        # 3. Fallback: Synthesize procedural PBR texture maps
        print(f"[PBREngine] No repository match >= {min_score} or download unavailable. Generating procedural fallback...")
        fallback_maps = self._generate_procedural_pbr_maps(
            component_id=component_id,
            category=category,
            target_dir=target_dir,
            target_color_hex=target_color_hex,
            roughness=estimated_roughness,
            metallic=estimated_metallic,
        )

        return {
            "component_id": component_id,
            "status": "procedural_synthesized",
            "source": "procedural_synthesis",
            "asset_id": f"procedural_{category}",
            "maps": fallback_maps,
            "target_color_hex": target_color_hex,
            "estimated_roughness": estimated_roughness,
            "estimated_metallic": estimated_metallic,
            "is_procedural_fallback": True,
        }

    def _download_polyhaven_asset(self, asset_id: str, target_dir: str, resolution: str = "1k") -> Optional[Dict[str, str]]:
        """Download and cache all maps for a PolyHaven texture."""
        cache_key = f"{asset_id}_{resolution}"
        if cache_key in self._session_asset_cache:
            print(f"[PBREngine] Using session-cached asset {asset_id} for new component.")
            resolved_maps = {}
            for map_type, src_path in self._session_asset_cache[cache_key].items():
                dest_file = os.path.join(target_dir, os.path.basename(src_path))
                shutil.copyfile(src_path, dest_file)
                resolved_maps[map_type] = dest_file
            return resolved_maps

        cached_asset_dir = os.path.join(self.cache_dir, f"polyhaven_{asset_id}_{resolution}")
        os.makedirs(cached_asset_dir, exist_ok=True)

        map_urls = self.polyhaven.get_map_urls(asset_id, resolution=resolution)
        if not map_urls:
            return None

        resolved_maps = {}
        
        def download_single_map(map_type: str, url: str) -> Optional[Tuple[str, str]]:
            ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
            cache_file = os.path.join(cached_asset_dir, f"{map_type}{ext}")

            # Download to cache if not already present
            if not os.path.isfile(cache_file) or os.path.getsize(cache_file) == 0:
                print(f"[PBREngine] Downloading {map_type} map: {url}")
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                try:
                    with urllib.request.urlopen(req, timeout=20) as resp, open(cache_file, "wb") as f_out:
                        shutil.copyfileobj(resp, f_out)
                except Exception as e:
                    print(f"[PBREngine] WARNING: Failed to download {map_type}: {e}")
                    return None

            # Copy to project target directory
            dest_file = os.path.join(target_dir, f"{map_type}{ext}")
            shutil.copyfile(cache_file, dest_file)
            return map_type, dest_file

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(download_single_map, m_type, url): m_type
                for m_type, url in map_urls.items()
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    resolved_maps[result[0]] = result[1]

        if resolved_maps:
            self._session_asset_cache[cache_key] = resolved_maps
            return resolved_maps
        return None

    def register_svbrdf_maps(
        self,
        component_id: str,
        svbrdf_manifest: Dict[str, Any],
        target_color_hex: Optional[str] = None,
        estimated_roughness: float = 0.5,
        estimated_metallic: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Directly registers locally-generated SVBRDF map bundle (from SVBRDFEngine.decompose_crop)
        as the component's resolved material, bypassing repository search.
        """
        maps = svbrdf_manifest.get("maps", {})
        return {
            "component_id": component_id,
            "status": "svbrdf_decomposed",
            "source": "svbrdf_engine",
            "asset_id": f"svbrdf_{component_id}",
            "maps": maps,
            "target_color_hex": target_color_hex,
            "estimated_roughness": estimated_roughness,
            "estimated_metallic": estimated_metallic,
            "is_procedural_fallback": False,
            "execution_tier": svbrdf_manifest.get("execution_tier", "tier_3_apu_cpu"),
            "normal_mode": svbrdf_manifest.get("normal_mode", "valid_tangent"),
            "composite_confidence_q": svbrdf_manifest.get("composite_confidence_q", 1.0),
        }

    def _validate_downloaded_maps(
        self,
        maps: Dict[str, str],
        target_dir: str,
        category: str,
        target_color_hex: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Runs quantitative acceptance gates on downloaded PBR maps.
        If a map fails, regenerates it using deterministic recovery (§ 1.5 & § 4.3).
        """
        # 1. Roughness gate: Var(R) >= 0.005
        rough_path = maps.get("roughness")
        if rough_path and os.path.isfile(rough_path):
            try:
                rough_img = cv2.imread(rough_path, cv2.IMREAD_GRAYSCALE)
                if rough_img is not None:
                    rough_float = rough_img.astype(np.float32) / 255.0
                    r_var = float(np.var(rough_float))
                    if r_var < 0.005:
                        print(f"[PBREngine] Downloaded roughness failed variance gate ({r_var:.5f} < 0.005). Applying procedural micro-facet perturbation.")
                        r_mean = float(np.mean(rough_float))
                        np.random.seed(42)
                        noise = np.random.randn(*rough_float.shape).astype(np.float32)
                        noise_blurred = cv2.GaussianBlur(noise, (5, 5), 1.0)
                        recovered_r = np.clip(r_mean + noise_blurred * 0.28, 0.02, 0.98)
                        cv2.imwrite(rough_path, (recovered_r * 255.0).astype(np.uint8))
            except Exception as e:
                print(f"[PBREngine] Warning during roughness gate validation: {e}")

        # 2. Normal flatness gate: Phi(N) <= 0.98
        norm_path = maps.get("normal")
        if norm_path and os.path.isfile(norm_path):
            try:
                norm_img = cv2.imread(norm_path, cv2.IMREAD_COLOR)
                if norm_img is not None:
                    norm_rgb = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
                    norm_float = (norm_rgb.astype(np.float32) / 127.5) - 1.0
                    flat_diff = np.sqrt(norm_float[:, :, 0]**2 + norm_float[:, :, 1]**2 + (norm_float[:, :, 2] - 1.0)**2)
                    flat_fraction = float(np.mean(flat_diff < 0.02))
                    if flat_fraction > 0.98:
                        print(f"[PBREngine] Downloaded normal map failed flatness gate ({flat_fraction:.3f} > 0.98). Recovering via Scharr frequency gradients.")
                        # Recover from diffuse map if available
                        diff_path = maps.get("diffuse")
                        if diff_path and os.path.isfile(diff_path):
                            diff_img = cv2.imread(diff_path, cv2.IMREAD_COLOR)
                            diff_rgb = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
                        else:
                            diff_rgb = np.full((norm_img.shape[0], norm_img.shape[1], 3), 128, dtype=np.uint8)

                        # High-pass Scharr recovery
                        gray = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                        low_pass = cv2.GaussianBlur(gray, (15, 15), 0)
                        h_high = gray - low_pass

                        gx = cv2.Scharr(h_high, cv2.CV_32F, 1, 0)
                        gy = cv2.Scharr(h_high, cv2.CV_32F, 0, 1)

                        if np.max(np.abs(gx)) < 1e-4 and np.max(np.abs(gy)) < 1e-4:
                            micro_grain = (np.random.randn(*gray.shape) * 0.05).astype(np.float32)
                            micro_grain = cv2.GaussianBlur(micro_grain, (5, 5), 1.0)
                            gx = cv2.Scharr(micro_grain, cv2.CV_32F, 1, 0)
                            gy = cv2.Scharr(micro_grain, cv2.CV_32F, 0, 1)

                        scale = 2.5
                        nx = -gx * scale
                        ny = -gy * scale
                        nz = np.ones_like(nx)
                        length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
                        nx, ny, nz = nx / length, ny / length, nz / length

                        rec_norm = np.zeros_like(norm_rgb)
                        rec_norm[:, :, 0] = np.clip(128.0 + 127.0 * nx, 0, 255).astype(np.uint8)
                        rec_norm[:, :, 1] = np.clip(128.0 + 127.0 * ny, 0, 255).astype(np.uint8)
                        rec_norm[:, :, 2] = np.clip(128.0 + 127.0 * nz, 0, 255).astype(np.uint8)

                        cv2.imwrite(norm_path, cv2.cvtColor(rec_norm, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[PBREngine] Warning during normal flatness validation: {e}")

        # 3. Albedo specular clipping gate: Gamma(A) <= 0.05
        diff_path = maps.get("diffuse")
        if diff_path and os.path.isfile(diff_path):
            try:
                diff_bgr = cv2.imread(diff_path, cv2.IMREAD_COLOR)
                if diff_bgr is not None:
                    diff_rgb = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2RGB)
                    lum = 0.2126 * (diff_rgb[:, :, 0] / 255.0) + 0.7152 * (diff_rgb[:, :, 1] / 255.0) + 0.0722 * (diff_rgb[:, :, 2] / 255.0)
                    clipped_fraction = float(np.mean(lum > 0.98))
                    if clipped_fraction > 0.05:
                        print(f"[PBREngine] Downloaded diffuse map failed specular clipping gate ({clipped_fraction:.3f} > 0.05). Inpainting specular highlights.")
                        spec_mask = (lum > 0.95).astype(np.uint8) * 255
                        inpainted_bgr = cv2.inpaint(diff_bgr, spec_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                        cv2.imwrite(diff_path, inpainted_bgr)
            except Exception as e:
                print(f"[PBREngine] Warning during albedo clipping validation: {e}")

        return maps

    def _generate_procedural_pbr_maps(
        self,
        component_id: str,
        category: str,
        target_dir: str,
        target_color_hex: Optional[str] = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        resolution: int = 1024,
    ) -> Dict[str, str]:
        """
        Generate physical PBR albedo, roughness, tangent-space normal, and metallic maps
        meeting quantitative acceptance thresholds (PIPELINE_SOLUTIONS_SPEC.md § 1.5 & § 4.3).
        Supports continuous physical metallic values and non-degenerate roughness.
        """
        os.makedirs(target_dir, exist_ok=True)

        # Parse target color or default
        if target_color_hex and target_color_hex.startswith("#") and len(target_color_hex) == 7:
            r = int(target_color_hex[1:3], 16)
            g = int(target_color_hex[3:5], 16)
            b = int(target_color_hex[5:7], 16)
        else:
            r, g, b = (128, 128, 128)

        base_rgb = np.array([r, g, b], dtype=np.int16)

        # 1. Diffuse / Albedo with physical micro-grain
        np.random.seed(42)
        noise = (np.random.randn(resolution, resolution, 3) * 6).astype(np.int16)
        diff_arr = np.clip(base_rgb + noise, 0, 255).astype(np.uint8)

        # Add directional grain if wood category
        if category.lower() in ["wood", "bamboo"]:
            grain = (np.sin(np.linspace(0, 30 * np.pi, resolution)) * 12).astype(np.int16)
            for y in range(resolution):
                diff_arr[y, :, :] = np.clip(diff_arr[y, :, :] + grain[y], 0, 255)

        diff_img = Image.fromarray(diff_arr)
        diff_path = os.path.join(target_dir, "diffuse.png")
        diff_img.save(diff_path)

        # 2. Roughness Map with strictly non-degenerate variance: Var(R) >= 0.005
        rough_val = float(np.clip(roughness, 0.05, 0.95))
        # Synthesize multi-scale procedural roughness noise
        noise_r = (np.random.randn(resolution, resolution) * 0.08).astype(np.float32)
        # Smooth with bilateral/Gaussian filter to create physical micro-facet distribution
        noise_r = cv2.GaussianBlur(noise_r, (7, 7), 1.5)
        rough_float = np.clip(rough_val + noise_r, 0.02, 0.98)
        # Verify Var(R) >= 0.005
        if np.var(rough_float) < 0.005:
            rough_float = np.clip(rough_float + (np.random.randn(resolution, resolution) * 0.06), 0.02, 0.98)

        rough_arr = (rough_float * 255.0).astype(np.uint8)
        rough_img = Image.fromarray(rough_arr)
        rough_path = os.path.join(target_dir, "roughness.png")
        rough_img.save(rough_path)

        # 3. Tangent-Space Normal Map via Scharr frequency gradients (eliminates flat [128, 128, 255])
        # Generate height field from luminance and procedural micro-relief
        gray_h = cv2.cvtColor(diff_arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # High-pass filter
        low_pass = cv2.GaussianBlur(gray_h, (15, 15), 0)
        h_high = gray_h - low_pass

        scale = 2.5
        gx = cv2.Scharr(h_high, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(h_high, cv2.CV_32F, 0, 1)

        # If perfectly flat (e.g. uniform color), inject subtle Voronoi/Perlin micro-grain
        if np.max(np.abs(gx)) < 1e-4 and np.max(np.abs(gy)) < 1e-4:
            micro_grain = (np.random.randn(resolution, resolution) * 0.05).astype(np.float32)
            micro_grain = cv2.GaussianBlur(micro_grain, (5, 5), 1.0)
            gx = cv2.Scharr(micro_grain, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(micro_grain, cv2.CV_32F, 0, 1)

        nx = -gx * scale
        ny = -gy * scale
        nz = np.ones_like(nx)
        length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6

        nx_norm = nx / length
        ny_norm = ny / length
        nz_norm = nz / length

        norm_arr = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        norm_arr[:, :, 0] = np.clip(128.0 + 127.0 * nx_norm, 0, 255).astype(np.uint8)
        norm_arr[:, :, 1] = np.clip(128.0 + 127.0 * ny_norm, 0, 255).astype(np.uint8)
        norm_arr[:, :, 2] = np.clip(128.0 + 127.0 * nz_norm, 0, 255).astype(np.uint8)

        norm_img = Image.fromarray(norm_arr)
        norm_path = os.path.join(target_dir, "normal.png")
        norm_img.save(norm_path)

        # 4. Metallic Map
        # Support continuous physical metallic values [0.0, 1.0]
        if metallic is not None and metallic > 0.0:
            metal_val = int(np.clip(metallic * 255.0, 0, 255))
        elif category.lower() in ["metal", "aluminum", "steel", "brass", "copper", "magnesium", "alloy", "chrome"]:
            metal_val = 220
        else:
            metal_val = 0
        metal_arr = np.full((resolution, resolution), metal_val, dtype=np.uint8)
        metal_path = os.path.join(target_dir, "metallic.png")
        Image.fromarray(metal_arr).save(metal_path)

        return {
            "diffuse": diff_path.replace("\\", "/"),
            "roughness": rough_path.replace("\\", "/"),
            "normal": norm_path.replace("\\", "/"),
            "metallic": metal_path.replace("\\", "/")
        }
