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
from PIL import Image


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

        url = f"{self.BASE_API}/assets?t=textures"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                self._assets_cache = json.loads(resp.read().decode("utf-8"))
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

        self.polyhaven = PolyHavenClient()
        self.ambientcg = AmbientCGClient()

    def resolve_material(
        self,
        component_id: str,
        keywords: List[str],
        category: str,
        target_dir: str,
        target_color_hex: Optional[str] = None,
        estimated_roughness: float = 0.5,
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
            print(f"[PBREngine] Selected '{asset_id}' from PolyHaven (Score: {best_match['score']:.1f})")
            downloaded = self._download_polyhaven_asset(asset_id, target_dir, resolution=resolution)
            if downloaded:
                return {
                    "component_id": component_id,
                    "status": "pbr_downloaded",
                    "source": "polyhaven",
                    "asset_id": asset_id,
                    "maps": downloaded,
                    "target_color_hex": target_color_hex,
                    "estimated_roughness": estimated_roughness,
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
        )

        return {
            "component_id": component_id,
            "status": "procedural_synthesized",
            "source": "procedural_synthesis",
            "asset_id": f"procedural_{category}",
            "maps": fallback_maps,
            "target_color_hex": target_color_hex,
            "estimated_roughness": estimated_roughness,
            "is_procedural_fallback": True,
        }

    def _download_polyhaven_asset(self, asset_id: str, target_dir: str, resolution: str = "1k") -> Optional[Dict[str, str]]:
        """Download and cache all maps for a PolyHaven texture."""
        cached_asset_dir = os.path.join(self.cache_dir, f"polyhaven_{asset_id}_{resolution}")
        os.makedirs(cached_asset_dir, exist_ok=True)

        map_urls = self.polyhaven.get_map_urls(asset_id, resolution=resolution)
        if not map_urls:
            return None

        resolved_maps = {}
        for map_type, url in map_urls.items():
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
                    continue

            # Copy to project target directory
            dest_file = os.path.join(target_dir, f"{map_type}{ext}")
            shutil.copyfile(cache_file, dest_file)
            resolved_maps[map_type] = dest_file

        return resolved_maps if resolved_maps else None

    def _generate_procedural_pbr_maps(
        self,
        component_id: str,
        category: str,
        target_dir: str,
        target_color_hex: Optional[str] = None,
        roughness: float = 0.5,
        resolution: int = 1024,
    ) -> Dict[str, str]:
        """Generate procedural albedo, roughness, and normal maps as fallback."""
        os.makedirs(target_dir, exist_ok=True)

        # Parse target color or default
        if target_color_hex and target_color_hex.startswith("#") and len(target_color_hex) == 7:
            r = int(target_color_hex[1:3], 16)
            g = int(target_color_hex[3:5], 16)
            b = int(target_color_hex[5:7], 16)
        else:
            r, g, b = (128, 128, 128)

        base_rgb = np.array([r, g, b], dtype=np.int16)

        # 1. Diffuse / Albedo with subtle micro-noise
        np.random.seed(42)
        noise = (np.random.randn(resolution, resolution, 3) * 4).astype(np.int16)
        diff_arr = np.clip(base_rgb + noise, 0, 255).astype(np.uint8)

        # Add directional grain if wood category
        if category.lower() in ["wood", "bamboo"]:
            grain = (np.sin(np.linspace(0, 30 * np.pi, resolution)) * 8).astype(np.int16)
            for y in range(resolution):
                diff_arr[y, :, :] = np.clip(diff_arr[y, :, :] + grain[y], 0, 255)

        diff_img = Image.fromarray(diff_arr)
        diff_path = os.path.join(target_dir, "diffuse.png")
        diff_img.save(diff_path)

        # 2. Roughness Map
        rough_val = int(max(0.0, min(1.0, roughness)) * 255)
        rough_arr = np.full((resolution, resolution), rough_val, dtype=np.int16)
        rough_noise = (np.random.randn(resolution, resolution) * 6).astype(np.int16)
        rough_arr = np.clip(rough_arr + rough_noise, 0, 255).astype(np.uint8)
        rough_img = Image.fromarray(rough_arr)
        rough_path = os.path.join(target_dir, "roughness.png")
        rough_img.save(rough_path)

        # 3. Flat / Subtle Normal Map (OpenGL standard: [128, 128, 255])
        norm_arr = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        norm_arr[:, :, 0] = 128
        norm_arr[:, :, 1] = 128
        norm_arr[:, :, 2] = 255
        norm_img = Image.fromarray(norm_arr)
        norm_path = os.path.join(target_dir, "normal.png")
        norm_img.save(norm_path)

        return {
            "diffuse": diff_path,
            "roughness": rough_path,
            "normal": norm_path,
        }
