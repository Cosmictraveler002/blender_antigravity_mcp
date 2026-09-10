"""
Multi-Viewport Renderer
=======================
Orchestrates capturing 14 calibrated camera angles of the active 3D model
inside Blender via the socket bridge.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List

from harness.config import DEFAULT_CONFIG, HarnessConfig, ViewportCaptureConfig
from harness.project_loader import ProjectDefinition
from harness.blender.client import send_blender_code


class MultiViewportRenderer:
    """Orchestrates 360-degree multi-angle render capture in Blender."""

    VIEWPORT_CONFIGS = [
        {"name": "front",          "azimuth": 0,   "elevation": 0,   "ortho": True},
        {"name": "back",           "azimuth": 180, "elevation": 0,   "ortho": True},
        {"name": "left",           "azimuth": 270, "elevation": 0,   "ortho": True},
        {"name": "right",          "azimuth": 90,  "elevation": 0,   "ortho": True},
        {"name": "top",            "azimuth": 0,   "elevation": 90,  "ortho": True},
        {"name": "bottom",         "azimuth": 0,   "elevation": -90, "ortho": True},
        {"name": "front_left_45",  "azimuth": 315, "elevation": 30,  "ortho": False},
        {"name": "front_right_45", "azimuth": 45,  "elevation": 30,  "ortho": False},
        {"name": "back_left_45",   "azimuth": 225, "elevation": 30,  "ortho": False},
        {"name": "back_right_45",  "azimuth": 135, "elevation": 30,  "ortho": False},
        {"name": "high_front",     "azimuth": 0,   "elevation": 60,  "ortho": False},
        {"name": "high_back",      "azimuth": 180, "elevation": 60,  "ortho": False},
        {"name": "low_front",      "azimuth": 0,   "elevation": -15, "ortho": False},
        {"name": "low_back",       "azimuth": 180, "elevation": -15, "ortho": False},
    ]

    def __init__(self, config: Optional[HarnessConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def capture_all(self, project: ProjectDefinition) -> Dict[str, Any]:
        """
        Executes multi-viewport capture for the given project.
        Renders 14 viewports to `project.viewports_dir`.

        Returns:
            Dict containing manifest of rendered files, metadata, and status.
        """
        os.makedirs(project.viewports_dir, exist_ok=True)

        # Determine viewport render settings
        vp_cfg = project.viewport or self.config.viewport
        render_cfg = project.render or self.config.render

        base_res = render_cfg.resolution
        scale = vp_cfg.resolution_scale
        vp_res_x = max(256, int(base_res[0] * scale))
        vp_res_y = max(256, int(base_res[1] * scale))
        samples = vp_cfg.samples
        padding = vp_cfg.padding_factor
        dist_factor = vp_cfg.camera_distance_factor

        print(f"[MultiViewportRenderer] Target Output: {project.viewports_dir}")
        print(f"[MultiViewportRenderer] Resolution: {vp_res_x}x{vp_res_y} (scale: {scale}) | Samples: {samples}")

        # Read blender capture script template
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "blender_scripts",
            "multi_viewport_render.py"
        )
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Blender viewport script not found: {script_path}")

        with open(script_path, "r", encoding="utf-8") as f:
            script_code = f.read()

        params = {
            "output_dir": project.viewports_dir.replace("\\", "/"),
            "resolution_x": vp_res_x,
            "resolution_y": vp_res_y,
            "samples": samples,
            "padding_factor": padding,
            "camera_distance_factor": dist_factor,
            "viewports": self.VIEWPORT_CONFIGS
        }

        # Build injection wrapper
        params_json_str = json.dumps(params)
        exec_payload = f"""
{script_code}

import json
params = json.loads({json.dumps(params_json_str)})
capture_result = run_viewport_capture(params)
"""

        host = self.config.blender.host
        port = self.config.blender.port

        print(f"[MultiViewportRenderer] Transmitting multi-viewport capture script to Blender ({host}:{port})...")
        res = send_blender_code(exec_payload, host=host, port=port)

        if not res or res.get("status") == "error":
            err_msg = res.get("message", "Unknown error from Blender") if res else "No response from Blender"
            raise RuntimeError(f"Multi-viewport capture failed: {err_msg}")

        # Verify renders on disk
        verified_viewports = []
        for vp in self.VIEWPORT_CONFIGS:
            name = vp["name"]
            expected_file = os.path.join(project.viewports_dir, f"viewport_{name}.png")
            if os.path.isfile(expected_file):
                verified_viewports.append({
                    "name": name,
                    "path": expected_file,
                    "exists": True,
                    "size_bytes": os.path.getsize(expected_file),
                    "azimuth": vp["azimuth"],
                    "elevation": vp["elevation"],
                    "ortho": vp["ortho"]
                })
            else:
                verified_viewports.append({
                    "name": name,
                    "path": expected_file,
                    "exists": False
                })

        manifest = {
            "project_name": project.name,
            "viewports_dir": project.viewports_dir,
            "total_viewports": len(self.VIEWPORT_CONFIGS),
            "captured_count": sum(1 for v in verified_viewports if v["exists"]),
            "resolution": [vp_res_x, vp_res_y],
            "viewports": verified_viewports,
            "blender_response": res
        }

        manifest_path = os.path.join(project.viewports_dir, "viewport_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[MultiViewportRenderer] Captured {manifest['captured_count']}/{len(self.VIEWPORT_CONFIGS)} viewports.")
        print(f"[MultiViewportRenderer] Saved manifest: {manifest_path}")

        return manifest
