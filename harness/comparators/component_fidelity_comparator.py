"""
Component Fidelity Comparator
=============================
Leverages the Gemini Vision API to compare multi-angle renders and component
close-ups of a generated 3D scene against the original reference image.
Focuses on extracting actionable refinement cues (geometry, textures, scale).
"""

import os
import json
import base64
import urllib.request
from typing import Dict, Any, Optional

COMPONENT_FIDELITY_SCHEMA = {
    "type": "object",
    "properties": {
        "component_name": {"type": "string"},
        "fidelity_score": {"type": "number", "description": "Score from 0 to 100 representing how closely this component matches the reference."},
        "missing_geometry_details": {
            "type": "array",
            "items": {"type": "string"}
        },
        "missing_texture_details": {
            "type": "array",
            "items": {"type": "string"}
        },
        "proportional_discrepancies": {
            "type": "array",
            "items": {"type": "string"}
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific, actionable steps to improve the component fidelity (e.g. 'Scale up the roof by 1.2x')."
        }
    },
    "required": ["component_name", "fidelity_score", "missing_geometry_details", "missing_texture_details", "proportional_discrepancies", "recommended_actions"]
}

class ComponentFidelityComparator:
    """Uses Gemini Vision to compare a component-specific render against the master reference image."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

    def compare(self, reference_image_path: str, render_image_path: str, component_name: str) -> Dict[str, Any]:
        """
        Compare a specific render against the reference image.
        """
        reference_image_path = os.path.abspath(reference_image_path)
        render_image_path = os.path.abspath(render_image_path)

        if not os.path.isfile(reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
        if not os.path.isfile(render_image_path):
            raise FileNotFoundError(f"Render image not found: {render_image_path}")

        print(f"[ComponentFidelity] Analyzing '{component_name}' fidelity vs reference image...")

        if not self.api_key:
            print("[ComponentFidelity] WARNING: No GEMINI_API_KEY found. Generating dummy report.")
            return self._generate_fallback_report(component_name)

        try:
            return self._call_gemini_api(reference_image_path, render_image_path, component_name)
        except Exception as e:
            print(f"[ComponentFidelity] Gemini API call failed: {e}")
            return self._generate_fallback_report(component_name)

    def _call_gemini_api(self, ref_img_path: str, render_img_path: str, component_name: str) -> Dict[str, Any]:
        """Query the Gemini REST API with two images."""
        def encode_img(path):
            with open(path, "rb") as f:
                img_bytes = f.read()
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            ext = os.path.splitext(path)[1].lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            return mime_type, b64_img

        mime_ref, b64_ref = encode_img(ref_img_path)
        mime_ren, b64_ren = encode_img(render_img_path)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        prompt = (
            f"You are an expert 3D artist reviewing a generated 3D scene. "
            f"The first image is the original reference photograph. The second image is a close-up/alternate angle render of the '{component_name}' from our generated 3D model. "
            f"Compare the '{component_name}' in the generated render to the corresponding element in the reference image. "
            f"Identify missing geometric details, missing texture details, and proportional discrepancies. "
            f"Provide actionable recommendations. "
            f"Respond ONLY with valid JSON matching this exact schema: "
            + json.dumps(COMPONENT_FIDELITY_SCHEMA)
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_ref,
                                "data": b64_ref
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_ren,
                                "data": b64_ren
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

        with urllib.request.urlopen(req, timeout=45) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

        text_content = resp_data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_content)

    def _generate_fallback_report(self, component_name: str) -> Dict[str, Any]:
        """Generate a fallback report if API fails or key is missing."""
        return {
            "component_name": component_name,
            "fidelity_score": 65.0,
            "missing_geometry_details": ["Fallback: Detail missing"],
            "missing_texture_details": ["Fallback: Texture missing"],
            "proportional_discrepancies": ["Fallback: Scale might be slightly off"],
            "recommended_actions": ["Fallback: Check scale and placement"]
        }
