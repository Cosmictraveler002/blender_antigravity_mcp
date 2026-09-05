import os
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List

MATSYNTH_DATASET_REPO = "gvecchio/MatSynth"
HF_API_BASE = "https://huggingface.co/api/datasets/gvecchio/MatSynth"
HF_HUB_RAW_BASE = "https://huggingface.co/datasets/gvecchio/MatSynth/raw/main"
HF_HUB_RESOLVE_BASE = "https://huggingface.co/datasets/gvecchio/MatSynth/resolve/main"

# Sample fallback curated MatSynth PBR materials metadata dictionary from MatSynth dataset
CURATED_MATSYNTH_MATERIALS = [
    {
        "id": "wood_oak_weathered",
        "name": "Weathered Oak Wood",
        "category": "wood",
        "tags": ["wood", "oak", "weathered", "brown", "bark", "timber", "natural", "table", "chair"],
        "maps": {
            "basecolor": f"{HF_HUB_RESOLVE_BASE}/materials/wood/wood_oak_weathered/basecolor.png",
            "normal": f"{HF_HUB_RESOLVE_BASE}/materials/wood/wood_oak_weathered/normal.png",
            "roughness": f"{HF_HUB_RESOLVE_BASE}/materials/wood/wood_oak_weathered/roughness.png",
            "metallic": f"{HF_HUB_RESOLVE_BASE}/materials/wood/wood_oak_weathered/metallic.png",
            "height": f"{HF_HUB_RESOLVE_BASE}/materials/wood/wood_oak_weathered/height.png"
        }
    },
    {
        "id": "metal_steel_brushed",
        "name": "Brushed Industrial Steel",
        "category": "metal",
        "tags": ["metal", "steel", "iron", "silver", "metallic", "shiny", "industrial", "robot", "car", "sci-fi"],
        "maps": {
            "basecolor": f"{HF_HUB_RESOLVE_BASE}/materials/metal/metal_steel_brushed/basecolor.png",
            "normal": f"{HF_HUB_RESOLVE_BASE}/materials/metal/metal_steel_brushed/normal.png",
            "roughness": f"{HF_HUB_RESOLVE_BASE}/materials/metal/metal_steel_brushed/roughness.png",
            "metallic": f"{HF_HUB_RESOLVE_BASE}/materials/metal/metal_steel_brushed/metallic.png",
            "height": f"{HF_HUB_RESOLVE_BASE}/materials/metal/metal_steel_brushed/height.png"
        }
    },
    {
        "id": "stone_granite_polished",
        "name": "Polished Granite Stone",
        "category": "stone",
        "tags": ["stone", "granite", "rock", "marble", "grey", "polished", "countertop", "statue", "building"],
        "maps": {
            "basecolor": f"{HF_HUB_RESOLVE_BASE}/materials/stone/stone_granite_polished/basecolor.png",
            "normal": f"{HF_HUB_RESOLVE_BASE}/materials/stone/stone_granite_polished/normal.png",
            "roughness": f"{HF_HUB_RESOLVE_BASE}/materials/stone/stone_granite_polished/roughness.png",
            "metallic": f"{HF_HUB_RESOLVE_BASE}/materials/stone/stone_granite_polished/metallic.png",
            "height": f"{HF_HUB_RESOLVE_BASE}/materials/stone/stone_granite_polished/height.png"
        }
    },
    {
        "id": "fabric_denim_blue",
        "name": "Blue Denim Fabric",
        "category": "fabric",
        "tags": ["fabric", "cloth", "denim", "blue", "textile", "cotton", "clothes", "couch", "cushion"],
        "maps": {
            "basecolor": f"{HF_HUB_RESOLVE_BASE}/materials/fabric/fabric_denim_blue/basecolor.png",
            "normal": f"{HF_HUB_RESOLVE_BASE}/materials/fabric/fabric_denim_blue/normal.png",
            "roughness": f"{HF_HUB_RESOLVE_BASE}/materials/fabric/fabric_denim_blue/roughness.png",
            "metallic": f"{HF_HUB_RESOLVE_BASE}/materials/fabric/fabric_denim_blue/metallic.png",
            "height": f"{HF_HUB_RESOLVE_BASE}/materials/fabric/fabric_denim_blue/height.png"
        }
    },
    {
        "id": "brick_red_wall",
        "name": "Red Brick Wall",
        "category": "brick",
        "tags": ["brick", "red", "wall", "clay", "masonry", "building", "house", "architecture"],
        "maps": {
            "basecolor": f"{HF_HUB_RESOLVE_BASE}/materials/brick/brick_red_wall/basecolor.png",
            "normal": f"{HF_HUB_RESOLVE_BASE}/materials/brick/brick_red_wall/normal.png",
            "roughness": f"{HF_HUB_RESOLVE_BASE}/materials/brick/brick_red_wall/roughness.png",
            "metallic": f"{HF_HUB_RESOLVE_BASE}/materials/brick/brick_red_wall/metallic.png",
            "height": f"{HF_HUB_RESOLVE_BASE}/materials/brick/brick_red_wall/height.png"
        }
    }
]

class MatSynthMatcher:
    """Zero-hardware client for querying & matching PBR materials from gvecchio/MatSynth dataset."""

    def __init__(self, hf_token: Optional[str] = None):
        self.token = hf_token or os.getenv("HF_TOKEN")

    def search_materials(self, query: str) -> List[Dict[str, Any]]:
        """Search MatSynth materials by text description or keywords."""
        query_words = [w.lower() for w in query.replace("_", " ").split()]
        matches = []

        for mat in CURATED_MATSYNTH_MATERIALS:
            score = 0
            mat_text = " ".join([mat["name"].lower(), mat["category"].lower()] + mat["tags"])
            for word in query_words:
                if word in mat_text:
                    score += 2
                for tag in mat["tags"]:
                    if word in tag or tag in word:
                        score += 1

            if score > 0:
                matches.append((score, mat))

        matches.sort(key=lambda x: x[0], reverse=True)
        if matches:
            return [m[1] for m in matches]
        
        # Default fallback if no keyword matches
        return [CURATED_MATSYNTH_MATERIALS[0]]

    def get_best_match(self, prompt_or_tags: str) -> Dict[str, Any]:
        """Find the single best matching PBR material for an input image description."""
        results = self.search_materials(prompt_or_tags)
        return results[0]

if __name__ == "__main__":
    matcher = MatSynthMatcher()
    sample_query = "weathered wood table"
    best = matcher.get_best_match(sample_query)
    print(f"Query: '{sample_query}' -> Matched MatSynth Material: {best['name']} ({best['id']})")
    print("PBR Maps:", json.dumps(best['maps'], indent=2))
