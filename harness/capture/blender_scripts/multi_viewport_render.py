"""
Multi-Viewport Render Script for Blender
=========================================
Executes inside Blender (via MCP socket or direct bpy execution).
Orbits the model across 14 calibrated viewpoints and renders each to disk.
"""

import bpy
import math
import os
import json
import mathutils


def run_viewport_capture(params=None):
    if params is None:
        params = {}

    output_dir = params.get("output_dir", os.path.abspath("./outputs/renders/viewports"))
    os.makedirs(output_dir, exist_ok=True)

    resolution_x = params.get("resolution_x", 960)
    resolution_y = params.get("resolution_y", 540)
    samples = params.get("samples", 64)
    padding_factor = params.get("padding_factor", 1.25)
    camera_distance_factor = params.get("camera_distance_factor", 3.2)

    # Standard 14-angle orbit configuration
    default_configs = [
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

    configs = params.get("viewports", default_configs)
    scene = bpy.context.scene

    # 1. Compute bounding box of all relevant mesh / geometry objects
    min_co = [float("inf")] * 3
    max_co = [float("-inf")] * 3
    has_geometry = False

    geo_types = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'GREASEPENCIL', 'GPENCIL'}

    # Exclude background studio floor / ground planes from character bounding box
    floor_keywords = ['floor', 'ground', 'backdrop', 'studio_floor']
    floor_objs = [
        obj for obj in scene.objects
        if obj.type == 'MESH' and any(k in obj.name.lower() for k in floor_keywords)
    ]

    candidate_objs = [
        obj for obj in scene.objects
        if obj.type in geo_types and not obj.hide_render and obj not in floor_objs
    ]
    if not candidate_objs:
        candidate_objs = [obj for obj in scene.objects if obj.type in geo_types]

    hidden_floor_objs = []
    for f_obj in floor_objs:
        if not f_obj.hide_render:
            f_obj.hide_render = True
            hidden_floor_objs.append(f_obj)

    for obj in candidate_objs:
        has_geometry = True
        try:
            for corner in obj.bound_box:
                world_pt = obj.matrix_world @ mathutils.Vector(corner)
                for i in range(3):
                    min_co[i] = min(min_co[i], world_pt[i])
                    max_co[i] = max(max_co[i], world_pt[i])
        except Exception:
            world_loc = obj.matrix_world.translation
            for i in range(3):
                min_co[i] = min(min_co[i], world_loc[i] - 1.0)
                max_co[i] = max(max_co[i], world_loc[i] + 1.0)

    if not has_geometry:
        min_co = [-1.0, -1.0, -1.0]
        max_co = [1.0, 1.0, 1.0]

    center = mathutils.Vector([(min_co[i] + max_co[i]) / 2.0 for i in range(3)])
    dims = mathutils.Vector([max(max_co[i] - min_co[i], 0.1) for i in range(3)])
    max_dim = max(dims.x, dims.y, dims.z)
    bbox_radius = 0.5 * math.sqrt(dims.x**2 + dims.y**2 + dims.z**2)
    bbox_radius = max(bbox_radius, 0.5)

    print(f"[MultiViewport] Center: {center}, Dims: {dims}, MaxDim: {max_dim:.2f}, Radius: {bbox_radius:.2f}")

    # 2. Store original scene settings
    orig_cam = scene.camera
    orig_res_x = scene.render.resolution_x
    orig_res_y = scene.render.resolution_y
    orig_res_pct = scene.render.resolution_percentage
    orig_filepath = scene.render.filepath
    orig_film_transparent = scene.render.film_transparent

    # Set viewport render resolution & format
    scene.render.resolution_x = int(resolution_x)
    scene.render.resolution_y = int(resolution_y)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    # Tune engine samples
    orig_cycles_samples = None
    if hasattr(scene, "cycles") and scene.render.engine == 'CYCLES':
        orig_cycles_samples = scene.cycles.samples
        scene.cycles.samples = min(scene.cycles.samples, samples)

    orig_eevee_samples = None
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            orig_eevee_samples = scene.eevee.taa_render_samples
            scene.eevee.taa_render_samples = min(scene.eevee.taa_render_samples, samples)

    # 3. Create dedicated capture camera
    cam_data = bpy.data.cameras.new(name="VP_Capture_Cam_Data")
    cam_obj = bpy.data.objects.new(name="VP_Capture_Cam", object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Create subtle camera headlight to prevent completely black shadows on rear/bottom
    headlight_data = bpy.data.lights.new(name="VP_Headlight_Data", type='POINT')
    headlight_data.energy = 50.0
    headlight_data.shadow_soft_size = 0.5
    headlight_obj = bpy.data.objects.new(name="VP_Headlight", object_data=headlight_data)
    scene.collection.objects.link(headlight_obj)
    headlight_obj.parent = cam_obj
    headlight_obj.location = (0, 0, 0.1)

    rendered_viewports = []

    try:
        min_persp_factor = (1.0 / math.tan(math.radians(13.5))) * padding_factor
        effective_dist_factor = max(camera_distance_factor, min_persp_factor)
        orbit_radius = bbox_radius * effective_dist_factor

        for cfg in configs:
            name = cfg["name"]
            az_deg = cfg["azimuth"]
            el_deg = cfg["elevation"]
            is_ortho = cfg.get("ortho", False)

            az = math.radians(az_deg)
            el = math.radians(el_deg)
            r_horiz = orbit_radius * math.cos(el)

            # Blender coordinate axes: +X = right, -Y = front, +Y = back, +Z = up
            cam_x = center.x + r_horiz * math.sin(az)
            cam_y = center.y - r_horiz * math.cos(az)
            cam_z = center.z + orbit_radius * math.sin(el)
            cam_obj.location = mathutils.Vector((cam_x, cam_y, cam_z))

            # Orient camera toward center
            direction = (center - cam_obj.location).normalized()
            if abs(direction.z) > 0.999:
                if direction.z < 0:
                    cam_obj.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0))
                else:
                    cam_obj.rotation_euler = mathutils.Euler((math.pi, 0.0, 0.0))
            else:
                quat = direction.to_track_quat('-Z', 'Y')
                cam_obj.rotation_euler = quat.to_euler()

            aspect_ratio = float(resolution_x) / max(float(resolution_y), 1.0)
            if is_ortho:
                cam_data.type = 'ORTHO'
                cam_data.sensor_fit = 'VERTICAL'
                cam_data.ortho_scale = max_dim * padding_factor * 1.25
            else:
                cam_data.type = 'PERSP'
                cam_data.lens = 50.0
                cam_data.sensor_fit = 'VERTICAL'

            out_filename = f"viewport_{name}.png"
            out_filepath = os.path.normpath(os.path.join(output_dir, out_filename))
            scene.render.filepath = out_filepath

            print(f"[MultiViewport] Rendering '{name}' (Az: {az_deg}°, El: {el_deg}°, Ortho: {is_ortho}) -> {out_filepath}")
            bpy.ops.render.render(write_still=True)

            rendered_viewports.append({
                "name": name,
                "file_path": out_filepath,
                "file_name": out_filename,
                "azimuth": az_deg,
                "elevation": el_deg,
                "ortho": is_ortho,
                "camera_location": [round(c, 3) for c in cam_obj.location],
                "target_center": [round(c, 3) for c in center]
            })

    finally:
        # 4. Clean up temporary objects & restore scene state
        if cam_obj.name in scene.collection.objects:
            scene.collection.objects.unlink(cam_obj)
        if headlight_obj.name in scene.collection.objects:
            scene.collection.objects.unlink(headlight_obj)

        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)
        bpy.data.objects.remove(headlight_obj, do_unlink=True)
        bpy.data.lights.remove(headlight_data, do_unlink=True)

        scene.camera = orig_cam
        scene.render.resolution_x = orig_res_x
        scene.render.resolution_y = orig_res_y
        scene.render.resolution_percentage = orig_res_pct
        scene.render.film_transparent = orig_film_transparent

        for f_obj in hidden_floor_objs:
            try:
                f_obj.hide_render = False
            except Exception:
                pass

        if orig_cycles_samples is not None and hasattr(scene, "cycles"):
            scene.cycles.samples = orig_cycles_samples
        if orig_eevee_samples is not None and hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = orig_eevee_samples

    manifest = {
        "status": "success",
        "total_viewports": len(rendered_viewports),
        "output_dir": output_dir,
        "resolution": [resolution_x, resolution_y],
        "bounding_box": {
            "center": [round(c, 3) for c in center],
            "dimensions": [round(d, 3) for d in dims],
            "radius": round(bbox_radius, 3)
        },
        "viewports": rendered_viewports
    }
    return manifest


if __name__ == "__main__":
    # If run standalone inside Blender
    result = run_viewport_capture()
    print("MultiViewport Result:", json.dumps(result, indent=2))
