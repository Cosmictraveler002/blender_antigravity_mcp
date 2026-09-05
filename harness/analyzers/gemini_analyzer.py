"""
Gemini Vision Semantic Analyzer
=================================
Extracts high-level semantic descriptors from reference imagery:
  - Component breakdown (categories, surface finishes, roughness, metallic)
  - PBR texture search keywords (for PolyHaven & ambientCG querying)
  - Typography & branding (text strings, fonts, orientation, placement)
  - Recommended studio lighting and environment HDRI

Dual-Execution Model:
  1. Agent/Interactive Mode: Reads pre-generated or agent-seeded vision analysis
     from projects/<project>/outputs/specs/gemini_vision_analysis.json.
  2. Standalone API Mode: Directly queries Google Gemini Vision REST API
     if GEMINI_API_KEY or GOOGLE_API_KEY is available.
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List


GEMINI_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "object_summary": {"type": "string"},
        "object_type": {"type": "string"},
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component_id": {"type": "string"},
                    "display_name": {"type": "string"},
                    "category": {"type": "string"},
                    "sub_category": {"type": "string"},
                    "visual_description": {"type": "string"},
                    "pbr_material_keywords": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "color_hex": {"type": "string"},
                    "estimated_roughness": {"type": "number"},
                    "estimated_metallic": {"type": "number"},
                    "normal_intensity": {"type": "number"}
                },
                "required": ["component_id", "category", "pbr_material_keywords", "color_hex"]
            }
        },
        "typography_and_labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_component": {"type": "string"},
                    "placement": {"type": "string"},
                    "orientation": {"type": "string"},
                    "font_style": {"type": "string"},
                    "color_hex": {"type": "string"},
                    "application_method": {"type": "string"}
                },
                "required": ["text", "target_component", "font_style"]
            }
        },
        "lighting_and_environment": {
            "type": "object",
            "properties": {
                "recommended_hdri_type": {"type": "string"},
                "recommended_hdri_keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "key_light_direction": {"type": "string"}
            }
        }
    },
    "required": ["object_summary", "components", "typography_and_labels"]
}


class GeminiVisionAnalyzer:
    """Multi-modal vision analyzer for extracting semantic material and textual descriptors."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

    def analyze(self, image_path: str, existing_spec_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze reference image or load existing agent-seeded analysis.

        Args:
            image_path: Path to the reference image.
            existing_spec_path: Optional path to pre-existing gemini_vision_analysis.json.

        Returns:
            Dict conforming to GEMINI_ANALYSIS_SCHEMA.
        """
        image_path = os.path.abspath(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        # 1. Check if agent-seeded analysis already exists
        if existing_spec_path and os.path.isfile(existing_spec_path):
            print(f"[GeminiVision] Loading existing vision analysis from: {existing_spec_path}")
            with open(existing_spec_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self._validate_analysis(data):
                return data

        # 2. If API key is available, call Gemini API
        if self.api_key:
            print(f"[GeminiVision] Calling Gemini API ({self.model_name}) for image analysis...")
            try:
                result = self._call_gemini_api(image_path)
                if result:
                    return result
            except Exception as e:
                print(f"[GeminiVision] WARNING: Gemini API call failed: {e}")

        # 3. Fallback: synthesize baseline descriptors from image
        print("[GeminiVision] Generating baseline semantic descriptors...")
        return self._generate_fallback_analysis(image_path)

    def _call_gemini_api(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Query the Gemini REST API with image and structured prompt."""
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        prompt = (
            "Analyze this 3D product reference photograph for automated 3D reconstruction in Blender. "
            "Identify all physical components, materials, surface finishes, PBR search keywords for PolyHaven/ambientCG, "
            "all visible text/branding typography with exact text, orientation, and font style, and studio lighting setup. "
            "Respond ONLY with valid JSON matching this exact schema: "
            + json.dumps(GEMINI_ANALYSIS_SCHEMA)
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_img
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

        text_content = resp_data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_content)

    def _validate_analysis(self, data: Dict[str, Any]) -> bool:
        """Quick check for required schema keys."""
        return "components" in data and "typography_and_labels" in data

    def _generate_fallback_analysis(self, image_path: str) -> Dict[str, Any]:
        """Generate baseline descriptors if no agent file or API key is present."""
        base_name = os.path.splitext(os.path.basename(image_path))[0].replace("_", " ")
        return {
            "object_summary": f"Reconstruction target: {base_name}",
            "object_type": "object",
            "components": [
                {
                    "component_id": "main_body",
                    "display_name": "Main Body",
                    "category": "metal",
                    "sub_category": "coated_metal",
                    "visual_description": "Matte finish main body structure",
                    "pbr_material_keywords": ["metal", "matte", "smooth"],
                    "color_hex": "#202022",
                    "estimated_roughness": 0.70,
                    "estimated_metallic": 0.0,
                    "normal_intensity": 0.2
                }
            ],
            "typography_and_labels": [],
            "lighting_and_environment": {
                "recommended_hdri_type": "studio",
                "recommended_hdri_keywords": ["studio", "white", "softbox"],
                "key_light_direction": "front_left"
            }
        }
