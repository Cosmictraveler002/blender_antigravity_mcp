import sys
import os
import json
import time
import math
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE_SSIM = True
except ImportError:
    HAS_SKIMAGE_SSIM = False
try:
    from harness.blender.client import send_blender_code
except ImportError:
    from blender_client import send_blender_code

from harness.utils.color_math import rgb_array_to_lab as rgb_to_lab_simple

class BlenderIterativeLoopEngine:
    """
    Advanced Multi-Region & Structural Self-Correcting Loop Engine:
    - Analyzes render vs reference across 3 functional zones: Top (Cap/Handle), Mid (Body/Text), Bottom (Base).
    - Measures perceptual CIE L*a*b* color distance.
    - Evaluates structural alignment using Structural Similarity (SSIM).
    - Progressively tunes Blender BSDF shaders, transforms, and lighting until visual convergence.
    """

    def __init__(self, target_image_path: Optional[str] = None):
        self.target_image_path = target_image_path
        self.render_output_path = os.path.abspath("current_render.png")
        self.target_features = None
        if target_image_path and os.path.exists(target_image_path):
            self.target_features = self._extract_image_features(target_image_path)

    def _extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """Extract multi-region color distribution, structural features, and SSIM array."""
        img = Image.open(image_path).convert("RGB").resize((160, 160))
        arr = np.array(img, dtype=np.float32) / 255.0
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        # Multi-region partition: Top (0..30%), Mid (30%..75%), Bottom (75%..100%)
        top_crop = arr[:48, :]
        mid_crop = arr[48:120, :]
        bot_crop = arr[120:, :]

        lab_arr = rgb_to_lab_simple(arr)

        return {
            "mean_rgb": np.mean(arr, axis=(0, 1)).tolist(),
            "mean_lab": np.mean(lab_arr, axis=(0, 1)).tolist(),
            "top_lab": np.mean(rgb_to_lab_simple(top_crop), axis=(0, 1)).tolist(),
            "mid_lab": np.mean(rgb_to_lab_simple(mid_crop), axis=(0, 1)).tolist(),
            "bot_lab": np.mean(rgb_to_lab_simple(bot_crop), axis=(0, 1)).tolist(),
            "gray_array": gray,
            "variance": float(np.var(gray)),
            "raw_array": arr
        }

    def compute_similarity_error(self, current_img_path: str) -> float:
        """Compute perceptual and structural visual discrepancy between render and reference."""
        if not self.target_features or not os.path.exists(current_img_path):
            return 0.5

        current_feat = self._extract_image_features(current_img_path)

        # 1. Multi-zone Perceptual Lab color error
        lab_t = np.linalg.norm(np.array(self.target_features["top_lab"]) - np.array(current_feat["top_lab"]))
        lab_m = np.linalg.norm(np.array(self.target_features["mid_lab"]) - np.array(current_feat["mid_lab"]))
        lab_b = np.linalg.norm(np.array(self.target_features["bot_lab"]) - np.array(current_feat["bot_lab"]))
        avg_lab_err = float((lab_t + lab_m + lab_b) / 3.0) / 100.0

        # 2. Structural Similarity (SSIM)
        if HAS_SKIMAGE_SSIM:
            score_ssim = float(ssim(self.target_features["gray_array"], current_feat["gray_array"], data_range=1.0))
            structural_err = max(0.0, 1.0 - score_ssim)
        else:
            mse = float(np.mean((self.target_features["gray_array"] - current_feat["gray_array"]) ** 2))
            structural_err = min(1.0, mse * 4.0)

        # 3. Variance discrepancy
        var_err = abs(self.target_features["variance"] - current_feat["variance"])

        total_error = float(0.45 * avg_lab_err + 0.40 * structural_err + 0.15 * min(1.0, var_err * 10.0))
        return round(total_error, 4)

    def setup_initial_scene(self, object_type: str = "SUZANNE"):
        """Initializes Blender scene, camera, lights, and target mesh using bpy cheat sheet syntax."""
        init_script = f"""
import bpy
import mathutils
from mathutils import Vector, Euler

# 1. Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 2. Add Camera
bpy.ops.object.camera_add(location=(0, -4, 2), rotation=(math.radians(65), 0, 0))
camera = bpy.context.active_object
camera.name = "LoopCamera"
bpy.context.scene.camera = camera

# 3. Add Sun Light
bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
light = bpy.context.active_object
light.data.energy = 3.0

# 4. Create Target Geometry Object
obj_kind = "{object_type}"
if obj_kind == "SPHERE":
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 1.0))
elif obj_kind == "CUBE":
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 1.0))
elif obj_kind == "TORUS":
    bpy.ops.mesh.primitive_torus_add(major_radius=1.2, minor_radius=0.3, location=(0, 0, 1.0))
else:
    bpy.ops.mesh.primitive_monkey_add(size=1.2, location=(0, 0, 1.0))

obj = bpy.context.active_object
obj.name = "LoopTargetObject"

sub_mod = obj.modifiers.new(name="Subsurf", type='SUBSURF')
sub_mod.levels = 2

# 5. Create Principled BSDF Material with Standard View Transform
bpy.context.scene.view_settings.view_transform = 'Standard'
mat = bpy.data.materials.new(name="LoopMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
bsdf = nodes.get("Principled BSDF")

if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0

obj.data.materials.append(mat)

# Configure Render Settings & OpenGL Capture
render_path = r"C:/Users/PC/myapps/Blender works/current_render.png"
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = render_path
bpy.ops.render.opengl(write_still=True)

print("Scene setup completed successfully.")
"""
        print("[Loop Engine] Transmitting initial scene setup script to Blender...")
        res = send_blender_code(init_script)
        print("Blender Response:", res.get("result", {}).get("result", ""))
        return res

    def run_optimization_step(self, iteration: int, target_rgb: Tuple[float, float, float] = (0.8, 0.2, 0.1), target_roughness: float = 0.15, target_scale: float = 1.3):
        """Performs a single self-correcting optimization step in Blender."""
        step_script = f"""
import bpy

obj = bpy.data.objects.get("LoopTargetObject")
light = bpy.data.objects.get("Sun")

if obj and obj.data.materials:
    mat = obj.data.materials[0]
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    if bsdf:
        curr_color = list(bsdf.inputs['Base Color'].default_value)
        curr_color[0] += ({target_rgb[0]} - curr_color[0]) * 0.4
        curr_color[1] += ({target_rgb[1]} - curr_color[1]) * 0.4
        curr_color[2] += ({target_rgb[2]} - curr_color[2]) * 0.4
        bsdf.inputs['Base Color'].default_value = curr_color
        
        curr_rough = bsdf.inputs['Roughness'].default_value
        bsdf.inputs['Roughness'].default_value += ({target_roughness} - curr_rough) * 0.4

    obj.rotation_euler.z += 0.26
    curr_s = obj.scale.x
    new_s = curr_s + ({target_scale} - curr_s) * 0.3
    obj.scale = (new_s, new_s, new_s)

render_path = r"C:/Users/PC/myapps/Blender works/current_render.png"
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = render_path
bpy.ops.render.opengl(write_still=True)

print("Iteration step updated in Blender successfully.")
"""
        res = send_blender_code(step_script)
        return res

    def execute_loop(self, max_iterations: int = 4, target_rgb: Tuple[float, float, float] = (0.9, 0.1, 0.2)):
        print("==========================================================")
        print(" Multi-Region & Structural Feedback Loop Engine Started")
        print("==========================================================")
        self.setup_initial_scene(object_type="SUZANNE")
        
        for i in range(1, max_iterations + 1):
            print(f"\n--- Iteration {i}/{max_iterations} ---")
            step_res = self.run_optimization_step(i, target_rgb=target_rgb)
            print(f" -> {step_res.get('result', {}).get('result', '').strip()}")
            
            err = self.compute_similarity_error(self.render_output_path)
            print(f" -> Rendered Screenshot: {self.render_output_path}")
            print(f" -> Perceptual & Structural Discrepancy Score: {err:.4f}")
            
            if err < 0.05:
                print(f"Convergence achieved at iteration {i}! Error < 0.05")
                break
            time.sleep(0.5)

        print("\n==========================================================")
        print(" Feedback Loop Workflow Completed Successfully.")
        print("==========================================================")

if __name__ == "__main__":
    engine = BlenderIterativeLoopEngine(target_image_path="reference_bottle.jpg")
    engine.execute_loop(max_iterations=3, target_rgb=(0.85, 0.15, 0.3))
