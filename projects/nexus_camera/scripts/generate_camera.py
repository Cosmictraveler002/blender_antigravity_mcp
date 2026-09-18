"""
Procedural 3D Generator: NEXUS Mirrorless Camera (35mm F/1.8)
============================================================
Procedurally generates a photorealistic 3D asset of the NEXUS mirrorless camera
with 35mm F/1.8 lens based on 2D multi-view design specifications.
Constructs precision geometry in Blender via bpy & bmesh.
"""

import os
import sys
import json
import math

# Standalone execution wrapper: If run directly by Python outside Blender,
# transmit self to Blender socket server via send_blender_code
if "bpy" not in sys.modules:
    import subprocess
    from harness.blender.client import send_blender_code

    script_path = os.path.abspath(__file__)
    print(f"[Generator] Executing {script_path} via Blender socket client...")
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()

    res = send_blender_code(code)
    print("[Generator] Blender Response:", res.get("status"))
    if res.get("status") == "error":
        print("[Generator] Error:", res.get("message"))
        sys.exit(1)
    sys.exit(0)

# -------------------------------------------------------------
# Inside Blender Environment
# -------------------------------------------------------------
import bpy
import bmesh
from mathutils import Vector, Euler, Matrix

# Resolve Paths
spec_dir = os.environ.get("HARNESS_SPEC_DIR", "")
renders_dir = os.environ.get("HARNESS_RENDER_DIR", "")
textures_dir = os.environ.get("HARNESS_TEXTURE_DIR", "")

if not renders_dir:
    project_root = r"C:\Users\PC\myapps\Blender works\projects\nexus_camera"
    renders_dir = os.environ.get("HARNESS_RENDER_DIR", os.path.join(project_root, "outputs", "renders"))
    textures_dir = os.environ.get("HARNESS_TEXTURE_DIR", os.path.join(project_root, "outputs", "textures"))
    spec_dir = os.environ.get("HARNESS_SPEC_DIR", os.path.join(project_root, "outputs", "specs"))

os.makedirs(renders_dir, exist_ok=True)

# Load Geometry Specs if available
geom_spec = {}
geom_json = os.environ.get("HARNESS_GEOM_JSON") or os.path.join(spec_dir, "geometry_design_doc.json")
if os.path.isfile(geom_json):
    try:
        with open(geom_json, "r", encoding="utf-8") as f:
            geom_spec = json.load(f)
    except Exception as e:
        print(f"[Generator] Warning loading geometry spec: {e}")

# Dimensions in millimeters (1 BU = 1 mm) - Calibrated to 0.75:1 aspect ratio
BODY_W = 134.0
BODY_H = 78.0
BODY_D = 46.0

GRIP_W = 34.0
GRIP_H = 78.0
GRIP_PROTRUSION = 24.0

PRISM_W = 34.0
PRISM_H = 19.0
PRISM_D = 38.0

LENS_DIAMETER = 65.0
LENS_LENGTH = 73.0
LENS_CX = 7.0   # Optical center offset relative to chassis origin
LENS_CZ = -4.0

LCD_W = 74.0
LCD_H = 50.0

def clean_scene():
    """Clear previous scene objects and materials."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for item in list(col):
            try:
                col.remove(item)
            except Exception:
                pass

def setup_render_settings():
    """Configure metric millimeters, Cycles renderer, and color management."""
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = 'MILLIMETERS'

    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
    except Exception:
        scene.cycles.device = 'CPU'

    scene.cycles.samples = 64
    scene.cycles.adaptive_threshold = 0.03
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.65
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 576

# -------------------------------------------------------------
# Procedural Geometry Helpers
# -------------------------------------------------------------
def create_beveled_box(name, width, depth, height, bevel_radius=2.0, segments=3):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector((width, depth, height)), verts=bm.verts)

    if bevel_radius > 0.01:
        edges = [e for e in bm.edges]
        bmesh.ops.bevel(bm, geom=edges, offset=bevel_radius, segments=segments, profile=0.5, affect='EDGES')

    mesh = bpy.data.meshes.new(name + "Mesh")
    bm.to_mesh(mesh)
    bm.free()

    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def create_cylinder(name, radius, depth, vertices=48, bevel_radius=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        segments=vertices,
        radius1=radius,
        radius2=radius,
        depth=depth
    )

    if bevel_radius > 0.01:
        cap_edges = [e for e in bm.edges if any(abs(v.co.z) > (depth * 0.48) for v in e.verts)]
        if cap_edges:
            bmesh.ops.bevel(bm, geom=cap_edges, offset=bevel_radius, segments=2, profile=0.5, affect='EDGES')

    mesh = bpy.data.meshes.new(name + "Mesh")
    bm.to_mesh(mesh)
    bm.free()

    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def create_decal_plane(name, width, height):
    bm = bmesh.new()
    v0 = bm.verts.new((-width / 2.0, 0.0, -height / 2.0))
    v1 = bm.verts.new((width / 2.0, 0.0, -height / 2.0))
    v2 = bm.verts.new((width / 2.0, 0.0, height / 2.0))
    v3 = bm.verts.new((-width / 2.0, 0.0, height / 2.0))
    face = bm.faces.new((v0, v1, v2, v3))

    uv_layer = bm.loops.layers.uv.verify()
    face.loops[0][uv_layer].uv = (0.0, 0.0)
    face.loops[1][uv_layer].uv = (1.0, 0.0)
    face.loops[2][uv_layer].uv = (1.0, 1.0)
    face.loops[3][uv_layer].uv = (0.0, 1.0)

    mesh = bpy.data.meshes.new(name + "Mesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

# -------------------------------------------------------------
# Material System
# -------------------------------------------------------------
class MaterialSystem:
    def __init__(self, tex_dir):
        self.tex_dir = tex_dir
        self.mats = {}
        self._build_materials()

    def _create_pbr(self, name, base_color_rgb, metallic=0.0, roughness=0.5, normal_map_path=None, normal_strength=0.5):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out_node = nodes.new(type='ShaderNodeOutputMaterial')
        out_node.location = (400, 0)

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        bsdf.inputs['Base Color'].default_value = (base_color_rgb[0], base_color_rgb[1], base_color_rgb[2], 1.0)
        bsdf.inputs['Metallic'].default_value = float(metallic)
        bsdf.inputs['Roughness'].default_value = float(roughness)
        links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])

        if normal_map_path and os.path.isfile(normal_map_path):
            img = bpy.data.images.load(normal_map_path, check_existing=True)
            img.colorspace_settings.name = 'Non-Color'
            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.image = img
            tex_node.extension = 'REPEAT'

            coord_node = nodes.new(type='ShaderNodeTexCoord')
            map_node = nodes.new(type='ShaderNodeMapping')
            map_node.inputs['Scale'].default_value = (0.15, 0.15, 0.15)
            links.new(coord_node.outputs['Object'], map_node.inputs['Vector'])
            links.new(map_node.outputs['Vector'], tex_node.inputs['Vector'])

            norm_node = nodes.new(type='ShaderNodeNormalMap')
            norm_node.inputs['Strength'].default_value = normal_strength

            links.new(tex_node.outputs['Color'], norm_node.inputs['Color'])
            links.new(norm_node.outputs['Normal'], bsdf.inputs['Normal'])

        self.mats[name] = mat
        return mat

    def _build_materials(self):
        decals_dir = os.path.join(self.tex_dir, "decals")

        # 1. Magnesium Chassis (Target #49474B, linear ~ 0.068)
        self._create_pbr("BodyMagnesium", (0.068, 0.065, 0.072), metallic=0.65, roughness=0.32)

        # 2. Leatherette Rubber Handgrip (Textured, High Roughness, Dielectric)
        leather_norm = os.path.join(decals_dir, "leatherette_normal.png")
        self._create_pbr("GripRubber", (0.025, 0.024, 0.026), metallic=0.0, roughness=0.75, normal_map_path=leather_norm, normal_strength=0.8)

        # 3. Anodized Aluminum (Target #ABA9AF / Satin brushed metal)
        self._create_pbr("AnodizedAluminum", (0.35, 0.35, 0.38), metallic=0.75, roughness=0.25)

        # 4. Knurled Dials / Rings
        self._create_pbr("DialKnurled", (0.030, 0.029, 0.032), metallic=0.92, roughness=0.35)

        # 5. Chrome / Stainless Steel (Mount flange, Strap lugs, Rails)
        self._create_pbr("ChromePolished", (0.85, 0.85, 0.87), metallic=1.0, roughness=0.10)

        # 6. Optical Glass Lens Element with AR Coating
        glass_mat = bpy.data.materials.new(name="LensGlass")
        glass_mat.use_nodes = True
        g_nodes = glass_mat.node_tree.nodes
        g_links = glass_mat.node_tree.links
        g_nodes.clear()
        g_out = g_nodes.new(type='ShaderNodeOutputMaterial')
        g_bsdf = g_nodes.new(type='ShaderNodeBsdfPrincipled')
        g_bsdf.inputs['Base Color'].default_value = (0.95, 0.98, 1.0, 1.0)
        g_bsdf.inputs['Roughness'].default_value = 0.02
        g_bsdf.inputs['IOR'].default_value = 1.54
        if 'Transmission Weight' in g_bsdf.inputs:
            g_bsdf.inputs['Transmission Weight'].default_value = 0.98
        elif 'Transmission' in g_bsdf.inputs:
            g_bsdf.inputs['Transmission'].default_value = 0.98
        if 'Coat Weight' in g_bsdf.inputs:
            g_bsdf.inputs['Coat Weight'].default_value = 0.85
            g_bsdf.inputs['Coat Roughness'].default_value = 0.02
            if 'Coat Tint' in g_bsdf.inputs:
                g_bsdf.inputs['Coat Tint'].default_value = (0.80, 1.0, 0.88, 1.0)
        g_links.new(g_bsdf.outputs['BSDF'], g_out.inputs['Surface'])
        self.mats["LensGlass"] = glass_mat

        # 7. LCD Display Screen with Menu UI
        lcd_mat = bpy.data.materials.new(name="LCDScreen")
        lcd_mat.use_nodes = True
        l_nodes = lcd_mat.node_tree.nodes
        l_links = lcd_mat.node_tree.links
        l_nodes.clear()
        l_out = l_nodes.new(type='ShaderNodeOutputMaterial')
        l_bsdf = l_nodes.new(type='ShaderNodeBsdfPrincipled')
        l_bsdf.inputs['Roughness'].default_value = 0.05

        lcd_tex = os.path.join(decals_dir, "lcd_menu_display.png")
        if os.path.isfile(lcd_tex):
            img = bpy.data.images.load(lcd_tex, check_existing=True)
            tex_node = l_nodes.new(type='ShaderNodeTexImage')
            tex_node.image = img
            l_links.new(tex_node.outputs['Color'], l_bsdf.inputs['Base Color'])
            if 'Emission Color' in l_bsdf.inputs:
                l_links.new(tex_node.outputs['Color'], l_bsdf.inputs['Emission Color'])
                l_bsdf.inputs['Emission Strength'].default_value = 0.5
        else:
            l_bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.06, 1.0)
        l_links.new(l_bsdf.outputs['BSDF'], l_out.inputs['Surface'])
        self.mats["LCDScreen"] = lcd_mat

        # 8. Decal / Prism Logo Material
        logo_mat = bpy.data.materials.new(name="PrismLogo")
        logo_mat.use_nodes = True
        p_nodes = logo_mat.node_tree.nodes
        p_links = logo_mat.node_tree.links
        p_nodes.clear()
        p_out = p_nodes.new(type='ShaderNodeOutputMaterial')
        p_bsdf = p_nodes.new(type='ShaderNodeBsdfPrincipled')
        p_bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.98, 1.0)
        p_bsdf.inputs['Metallic'].default_value = 0.0
        p_bsdf.inputs['Roughness'].default_value = 0.35

        nexus_decal = os.path.join(decals_dir, "nexus_logo_decal.png")
        if os.path.isfile(nexus_decal):
            d_img = bpy.data.images.load(nexus_decal, check_existing=True)
            d_tex = p_nodes.new(type='ShaderNodeTexImage')
            d_tex.image = d_img
            p_coord = p_nodes.new(type='ShaderNodeTexCoord')
            p_links.new(p_coord.outputs['UV'], d_tex.inputs['Vector'])
            p_links.new(d_tex.outputs['Alpha'], p_bsdf.inputs['Alpha'])

        p_links.new(p_bsdf.outputs['BSDF'], p_out.inputs['Surface'])
        self.mats["PrismLogo"] = logo_mat

        # 9. Lens Bezel Printed Ring Material
        lens_ring_mat = bpy.data.materials.new(name="LensTextRing")
        lens_ring_mat.use_nodes = True
        r_nodes = lens_ring_mat.node_tree.nodes
        r_links = lens_ring_mat.node_tree.links
        r_nodes.clear()
        r_out = r_nodes.new(type='ShaderNodeOutputMaterial')
        r_bsdf = r_nodes.new(type='ShaderNodeBsdfPrincipled')
        r_bsdf.inputs['Roughness'].default_value = 0.35
        r_bsdf.inputs['Metallic'].default_value = 0.85
        lens_tex = os.path.join(decals_dir, "lens_ring_decal.png")
        if os.path.isfile(lens_tex):
            l_img = bpy.data.images.load(lens_tex, check_existing=True)
            l_t = r_nodes.new(type='ShaderNodeTexImage')
            l_t.image = l_img
            r_coord = r_nodes.new(type='ShaderNodeTexCoord')
            r_links.new(r_coord.outputs['Generated'], l_t.inputs['Vector'])
            r_links.new(l_t.outputs['Color'], r_bsdf.inputs['Base Color'])
        else:
            r_bsdf.inputs['Base Color'].default_value = (0.04, 0.04, 0.05, 1.0)
        r_links.new(r_bsdf.outputs['BSDF'], r_out.inputs['Surface'])
        self.mats["LensTextRing"] = lens_ring_mat

        # 10. Greek Alpha (α) Emblem Badge Material
        alpha_mat = bpy.data.materials.new(name="AlphaEmblem")
        alpha_mat.use_nodes = True
        a_nodes = alpha_mat.node_tree.nodes
        a_links = alpha_mat.node_tree.links
        a_nodes.clear()
        a_out = a_nodes.new(type='ShaderNodeOutputMaterial')
        a_bsdf = a_nodes.new(type='ShaderNodeBsdfPrincipled')
        a_bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.98, 1.0)
        a_bsdf.inputs['Metallic'].default_value = 0.85
        a_bsdf.inputs['Roughness'].default_value = 0.20
        alpha_tex = os.path.join(decals_dir, "alpha_emblem.png")
        if os.path.isfile(alpha_tex):
            a_img = bpy.data.images.load(alpha_tex, check_existing=True)
            a_t = a_nodes.new(type='ShaderNodeTexImage')
            a_t.image = a_img
            a_coord = a_nodes.new(type='ShaderNodeTexCoord')
            a_links.new(a_coord.outputs['UV'], a_t.inputs['Vector'])
            a_links.new(a_t.outputs['Alpha'], a_bsdf.inputs['Alpha'])
        a_links.new(a_bsdf.outputs['BSDF'], a_out.inputs['Surface'])
        self.mats["AlphaEmblem"] = alpha_mat

        # 11. Ribbed Focus Ring Material (Tactile rubber grip bands)
        f_mat = bpy.data.materials.new(name="FocusRing")
        f_mat.use_nodes = True
        f_nodes = f_mat.node_tree.nodes
        f_links = f_mat.node_tree.links
        f_nodes.clear()
        f_out = f_nodes.new(type='ShaderNodeOutputMaterial')
        f_bsdf = f_nodes.new(type='ShaderNodeBsdfPrincipled')
        f_bsdf.inputs['Base Color'].default_value = (0.025, 0.024, 0.027, 1.0)
        f_bsdf.inputs['Metallic'].default_value = 0.10
        f_bsdf.inputs['Roughness'].default_value = 0.65

        f_tex = f_nodes.new(type='ShaderNodeTexWave')
        f_tex.wave_type = 'BANDS'
        f_tex.bands_direction = 'X'
        f_tex.inputs['Scale'].default_value = 0.35
        f_bump = f_nodes.new(type='ShaderNodeBump')
        f_bump.inputs['Strength'].default_value = 0.80
        f_bump.inputs['Distance'].default_value = 0.04
        f_coord = f_nodes.new(type='ShaderNodeTexCoord')
        f_links.new(f_coord.outputs['Object'], f_tex.inputs['Vector'])
        f_links.new(f_tex.outputs['Color'], f_bump.inputs['Height'])
        f_links.new(f_bump.outputs['Normal'], f_bsdf.inputs['Normal'])
        f_links.new(f_bsdf.outputs['BSDF'], f_out.inputs['Surface'])
        self.mats["FocusRing"] = f_mat

# -------------------------------------------------------------
# Procedural Camera Builder
# -------------------------------------------------------------
class CameraBuilder:
    def __init__(self, materials: MaterialSystem):
        self.m = materials.mats

    def build_all(self):
        self.build_chassis()
        self.build_pentaprism_viewfinder()
        self.build_handgrip()
        self.build_lens_assembly()
        self.build_top_controls()
        self.build_rear_assembly()
        self.build_hardware()

    def build_chassis(self):
        """Main camera body chassis with chamfers."""
        chassis = create_beveled_box("CameraChassis", BODY_W, BODY_D, BODY_H, bevel_radius=3.0, segments=4)
        chassis.location = (0, 0, 0)
        chassis.data.materials.append(self.m["BodyMagnesium"])

        # Base plate trim
        base_trim = create_beveled_box("BasePlateTrim", BODY_W, BODY_D, 3.0, bevel_radius=0.8, segments=2)
        base_trim.location = (0, 0, -BODY_H / 2.0 + 1.5)
        base_trim.data.materials.append(self.m["AnodizedAluminum"])

        # Greek letter Alpha (α) chrome badge (viewer's right in front view)
        alpha_badge = create_decal_plane("AlphaEmblemBadge", 7.5, 7.5)
        alpha_badge.location = (LENS_CX + 34.0, -BODY_D / 2.0 - 0.2, BODY_H / 2.0 - 12.0)
        alpha_badge.data.materials.append(self.m["AlphaEmblem"])

        # Red tally / AF illuminator sensor window
        af_sensor = create_cylinder("AFAssistWindow", radius=2.5, depth=1.5, vertices=16, bevel_radius=0.2)
        af_sensor.rotation_euler = Euler((math.radians(90.0), 0, 0))
        af_sensor.location = (LENS_CX + 34.0, -BODY_D / 2.0 - 0.6, BODY_H / 2.0 - 22.0)
        af_sensor.data.materials.append(self.m["DialKnurled"])

    def build_pentaprism_viewfinder(self):
        """Faceted SLR-style pentaprism housing on top plate."""
        bm = bmesh.new()
        # Trapezoidal prism
        w_bot = PRISM_W
        w_top = PRISM_W * 0.72
        d_bot = PRISM_D
        d_top = PRISM_D * 0.65
        h = PRISM_H

        v1 = bm.verts.new((-w_bot/2, -d_bot/2, 0))
        v2 = bm.verts.new((w_bot/2, -d_bot/2, 0))
        v3 = bm.verts.new((w_bot/2, d_bot/2, 0))
        v4 = bm.verts.new((-w_bot/2, d_bot/2, 0))

        v5 = bm.verts.new((-w_top/2, -d_top/2, h))
        v6 = bm.verts.new((w_top/2, -d_top/2, h))
        v7 = bm.verts.new((w_top/2, d_top/2, h))
        v8 = bm.verts.new((-w_top/2, d_top/2, h))

        bm.faces.new((v4, v3, v2, v1))  # Bottom
        bm.faces.new((v5, v6, v7, v8))  # Top
        bm.faces.new((v1, v2, v6, v5))  # Front slope
        bm.faces.new((v2, v3, v7, v6))  # Right slope
        bm.faces.new((v3, v4, v8, v7))  # Back slope
        bm.faces.new((v4, v1, v5, v8))  # Left slope

        edges = [e for e in bm.edges]
        bmesh.ops.bevel(bm, geom=edges, offset=1.2, segments=3, profile=0.5, affect='EDGES')

        mesh = bpy.data.meshes.new("PentaprismMesh")
        bm.to_mesh(mesh)
        bm.free()

        for poly in mesh.polygons:
            poly.use_smooth = True

        prism = bpy.data.objects.new("PentaprismViewfinder", mesh)
        bpy.context.collection.objects.link(prism)
        prism.location = (LENS_CX, -2.0, BODY_H / 2.0)
        prism.data.materials.append(self.m["BodyMagnesium"])

        # Dedicated front NEXUS branding plate flush with the angled slope
        logo_plate = create_decal_plane("NexusLogoPlate", PRISM_W * 0.70, PRISM_H * 0.45)
        logo_plate.rotation_euler = Euler((math.radians(18.5), 0, 0))
        logo_plate.location = (LENS_CX, -2.0 - (d_bot + d_top) / 4.0 - 0.15, BODY_H / 2.0 + PRISM_H * 0.48)
        logo_plate.data.materials.append(self.m["PrismLogo"])

        # Hot shoe assembly on top of prism
        shoe_base = create_beveled_box("HotShoeBase", 18.0, 19.0, 3.2, bevel_radius=0.5)
        shoe_base.location = (LENS_CX, 0.0, BODY_H / 2.0 + PRISM_H + 1.2)
        shoe_base.data.materials.append(self.m["AnodizedAluminum"])

        # Hot shoe chrome side rails
        rail_l = create_beveled_box("HotShoeRailL", 2.2, 18.0, 1.8, bevel_radius=0.2)
        rail_l.location = (LENS_CX - 8.0, 0.0, BODY_H / 2.0 + PRISM_H + 2.8)
        rail_l.data.materials.append(self.m["ChromePolished"])

        rail_r = create_beveled_box("HotShoeRailR", 2.2, 18.0, 1.8, bevel_radius=0.2)
        rail_r.location = (LENS_CX + 8.0, 0.0, BODY_H / 2.0 + PRISM_H + 2.8)
        rail_r.data.materials.append(self.m["ChromePolished"])

        # Viewfinder soft rubber eyecup on rear (+Y)
        eyecup = create_cylinder("EyecupOval", radius=11.0, depth=6.0, vertices=32, bevel_radius=1.5)
        eyecup.rotation_euler = Euler((math.radians(90.0), 0, 0))
        eyecup.scale = Vector((1.2, 0.8, 1.0))
        eyecup.location = (LENS_CX, BODY_D / 2.0 + 3.0, BODY_H / 2.0 + PRISM_H * 0.45)
        eyecup.data.materials.append(self.m["GripRubber"])

    def build_handgrip(self):
        """Contoured ergonomic right-hand grip with leatherette texture."""
        grip_x = -BODY_W / 2.0 + GRIP_W / 2.0 - 2.0
        grip_y = -BODY_D / 2.0 - GRIP_PROTRUSION / 2.0 + 3.0
        grip_z = 0.0

        # Ergonomic curved grip body flush with top plate
        grip = create_beveled_box("HandgripBody", GRIP_W, GRIP_PROTRUSION + 6.0, BODY_H, bevel_radius=6.0, segments=5)
        grip.location = (grip_x, grip_y, grip_z)
        grip.data.materials.append(self.m["GripRubber"])

        # Front command dial embedded in grip top
        front_dial = create_cylinder("FrontCommandDial", radius=8.5, depth=4.5, vertices=32, bevel_radius=0.6)
        front_dial.rotation_euler = Euler((math.radians(20.0), 0, 0))
        front_dial.location = (grip_x, -BODY_D / 2.0 - GRIP_PROTRUSION * 0.8, BODY_H / 2.0 - 12.0)
        front_dial.data.materials.append(self.m["DialKnurled"])

    def build_lens_assembly(self):
        """Precision 35mm F/1.8 multi-element lens with ribbed focus ring & glass element."""
        mount_y = -BODY_D / 2.0

        # 1. Chrome Bayonet Mount Flange
        mount_flange = create_cylinder("LensMountFlange", radius=34.0, depth=4.0, vertices=48, bevel_radius=0.4)
        mount_flange.rotation_euler = Euler((math.radians(90.0), 0, 0))
        mount_flange.location = (LENS_CX, mount_y - 2.0, LENS_CZ)
        mount_flange.data.materials.append(self.m["ChromePolished"])

        # 2. Outer Mount Collar (Black Aluminum)
        mount_collar = create_cylinder("LensMountCollar", radius=33.2, depth=8.0, vertices=48, bevel_radius=0.6)
        mount_collar.rotation_euler = Euler((math.radians(90.0), 0, 0))
        mount_collar.location = (LENS_CX, mount_y - 8.0, LENS_CZ)
        mount_collar.data.materials.append(self.m["AnodizedAluminum"])

        # 3. Base Barrel Section (Fixed)
        barrel_base = create_cylinder("LensBarrelBase", radius=32.5, depth=14.0, vertices=48, bevel_radius=0.5)
        barrel_base.rotation_euler = Euler((math.radians(90.0), 0, 0))
        barrel_base.location = (LENS_CX, mount_y - 19.0, LENS_CZ)
        barrel_base.data.materials.append(self.m["AnodizedAluminum"])

        # 4. Broad Ribbed Manual Focus Ring
        focus_ring = create_cylinder("LensFocusRing", radius=33.0, depth=26.0, vertices=64, bevel_radius=0.8)
        focus_ring.rotation_euler = Euler((math.radians(90.0), 0, 0))
        focus_ring.location = (LENS_CX, mount_y - 39.0, LENS_CZ)
        focus_ring.data.materials.append(self.m["FocusRing"])

        # 5. Front Barrel Section with Filter Thread
        barrel_front = create_cylinder("LensBarrelFront", radius=32.5, depth=18.0, vertices=48, bevel_radius=0.5)
        barrel_front.rotation_euler = Euler((math.radians(90.0), 0, 0))
        barrel_front.location = (LENS_CX, mount_y - 61.0, LENS_CZ)
        barrel_front.data.materials.append(self.m["AnodizedAluminum"])

        # 6. Front Retaining Ring Bezel (Hollow tube with open aperture)
        bm_bezel = bmesh.new()
        bmesh.ops.create_cone(
            bm_bezel,
            cap_ends=False,
            segments=48,
            radius1=32.2,
            radius2=32.2,
            depth=4.0
        )
        mesh_bezel = bpy.data.meshes.new("LensFrontBezelMesh")
        bm_bezel.to_mesh(mesh_bezel)
        bm_bezel.free()
        front_bezel = bpy.data.objects.new("LensFrontBezel", mesh_bezel)
        bpy.context.collection.objects.link(front_bezel)
        mod_sol = front_bezel.modifiers.new(name="Solidify", type='SOLIDIFY')
        mod_sol.thickness = 1.4
        front_bezel.rotation_euler = Euler((math.radians(90.0), 0, 0))
        front_bezel.location = (LENS_CX, mount_y - 71.5, LENS_CZ)
        front_bezel.data.materials.append(self.m["AnodizedAluminum"])

        # 7. Printed Lens Spec Bezel Ring ("35mm F/1.8 Ø55")
        text_ring = create_cylinder("LensTextRing", radius=30.8, depth=0.2, vertices=48)
        text_ring.rotation_euler = Euler((math.radians(90.0), 0, 0))
        text_ring.location = (LENS_CX, mount_y - 70.0, LENS_CZ)
        text_ring.data.materials.append(self.m["LensTextRing"])

        # 8. Internal Aperture Diaphragm & Optical Chamber
        iris_obj = create_cylinder("LensApertureIris", radius=22.0, depth=1.5, vertices=32, bevel_radius=0.1)
        iris_obj.rotation_euler = Euler((math.radians(90.0), 0, 0))
        iris_obj.location = (LENS_CX, mount_y - 50.0, LENS_CZ)
        iris_obj.data.materials.append(self.m["DialKnurled"])

        # Inner aperture hole disk
        ap_hole = create_cylinder("LensApertureHole", radius=11.0, depth=1.6, vertices=32, bevel_radius=0.0)
        ap_hole.rotation_euler = Euler((math.radians(90.0), 0, 0))
        ap_hole.location = (LENS_CX, mount_y - 50.0, LENS_CZ)
        mat_sensor = bpy.data.materials.new(name="SensorBlack")
        mat_sensor.use_nodes = True
        bsdf_s = mat_sensor.node_tree.nodes.get("Principled BSDF")
        if bsdf_s:
            bsdf_s.inputs["Base Color"].default_value = (0.01, 0.01, 0.015, 1.0)
            bsdf_s.inputs["Roughness"].default_value = 0.90
        ap_hole.data.materials.append(mat_sensor)

        # 9. Optical Front Glass Element (True convex meniscus)
        bm_glass = bmesh.new()
        bmesh.ops.create_uvsphere(bm_glass, u_segments=48, v_segments=24, radius=23.0)
        del_verts = [v for v in bm_glass.verts if v.co.z < 13.0]
        bmesh.ops.delete(bm_glass, geom=del_verts, context='VERTS')
        bmesh.ops.scale(bm_glass, vec=Vector((1.0, 1.0, 0.40)), verts=bm_glass.verts)
        min_z = min(v.co.z for v in bm_glass.verts)
        for v in bm_glass.verts:
            v.co.z -= min_z
        boundary_edges = [e for e in bm_glass.edges if len(e.link_faces) == 1]
        if boundary_edges:
            bmesh.ops.edgeloop_fill(bm_glass, edges=boundary_edges)

        mesh_glass = bpy.data.meshes.new("LensGlassMesh")
        bm_glass.to_mesh(mesh_glass)
        bm_glass.free()
        for p in mesh_glass.polygons:
            p.use_smooth = True

        glass_obj = bpy.data.objects.new("LensFrontElement", mesh_glass)
        bpy.context.collection.objects.link(glass_obj)
        glass_obj.rotation_euler = Euler((math.radians(-90.0), 0, 0))
        glass_obj.location = (LENS_CX, mount_y - 67.5, LENS_CZ)
        glass_obj.data.materials.append(self.m["LensGlass"])

    def build_top_controls(self):
        """Top deck dials: Mode dial, Exposure dial, Shutter collar, C1/C2."""
        top_z = BODY_H / 2.0

        # 1. Mode Dial with Center Lock Button (Camera Left / Top View Left)
        mode_x = -BODY_W * 0.32
        mode_dial = create_cylinder("ModeDial", radius=11.5, depth=7.5, vertices=32, bevel_radius=0.5)
        mode_dial.location = (mode_x, -2.0, top_z + 3.75)
        mode_dial.data.materials.append(self.m["DialKnurled"])

        mode_lock = create_cylinder("ModeDialLock", radius=4.0, depth=2.0, vertices=24, bevel_radius=0.2)
        mode_lock.location = (mode_x, -2.0, top_z + 8.5)
        mode_lock.data.materials.append(self.m["ChromePolished"])

        # 2. Exposure Compensation Dial (Camera Right)
        exp_x = BODY_W * 0.38
        exp_dial = create_cylinder("ExposureCompensationDial", radius=10.5, depth=6.5, vertices=32, bevel_radius=0.5)
        exp_dial.location = (exp_x, 4.0, top_z + 3.25)
        exp_dial.data.materials.append(self.m["DialKnurled"])

        # 3. Shutter Release Button with Power ON/OFF Collar Lever
        shut_x = -BODY_W * 0.35
        shut_y = -BODY_D / 2.0 - GRIP_PROTRUSION * 0.4

        # Power collar switch
        power_collar = create_cylinder("PowerSwitchCollar", radius=8.2, depth=3.5, vertices=24, bevel_radius=0.4)
        power_collar.location = (shut_x, shut_y, top_z + 1.75)
        power_collar.data.materials.append(self.m["AnodizedAluminum"])

        # Power switch toggle lever
        power_lever = create_beveled_box("PowerSwitchLever", 3.0, 6.5, 3.2, bevel_radius=0.3)
        power_lever.location = (shut_x - 7.0, shut_y - 2.0, top_z + 2.0)
        power_lever.data.materials.append(self.m["AnodizedAluminum"])

        # Shutter button
        shutter_btn = create_cylinder("ShutterButton", radius=5.8, depth=4.0, vertices=24, bevel_radius=0.3)
        shutter_btn.location = (shut_x, shut_y, top_z + 4.5)
        shutter_btn.data.materials.append(self.m["ChromePolished"])

        # 4. Custom Buttons C1 & C2
        c1_btn = create_cylinder("ButtonC1", radius=3.2, depth=1.8, vertices=16, bevel_radius=0.2)
        c1_btn.location = (exp_x - 16.0, 8.0, top_z + 1.0)
        c1_btn.data.materials.append(self.m["AnodizedAluminum"])

        c2_btn = create_cylinder("ButtonC2", radius=3.2, depth=1.8, vertices=16, bevel_radius=0.2)
        c2_btn.location = (exp_x - 16.0, -4.0, top_z + 1.0)
        c2_btn.data.materials.append(self.m["AnodizedAluminum"])

    def build_rear_assembly(self):
        """Articulated LCD monitor screen and control buttons."""
        rear_y = BODY_D / 2.0

        # 1. Articulated LCD Monitor Frame
        lcd_frame = create_beveled_box("LCDMonitorFrame", LCD_W, 3.5, LCD_H, bevel_radius=1.5, segments=3)
        lcd_frame.location = (-BODY_W * 0.14, rear_y + 1.75, -2.0)
        lcd_frame.data.materials.append(self.m["AnodizedAluminum"])

        # LCD Display Panel
        lcd_display = create_beveled_box("LCDDisplayPanel", LCD_W - 5.0, 0.2, LCD_H - 5.0, bevel_radius=0.2)
        lcd_display.location = (-BODY_W * 0.14, rear_y + 3.6, -2.0)
        lcd_display.data.materials.append(self.m["LCDScreen"])

        # 2. Multi-Selector Joystick (+X side of rear)
        joy_x = BODY_W * 0.32
        joystick = create_cylinder("RearJoystick", radius=4.0, depth=5.0, vertices=20, bevel_radius=0.5)
        joystick.rotation_euler = Euler((math.radians(90.0), 0, 0))
        joystick.location = (joy_x, rear_y + 2.5, 16.0)
        joystick.data.materials.append(self.m["DialKnurled"])

        # 3. Rear Command Dial
        rear_dial = create_cylinder("RearCommandDial", radius=9.0, depth=3.5, vertices=32, bevel_radius=0.4)
        rear_dial.rotation_euler = Euler((math.radians(90.0), 0, 0))
        rear_dial.location = (joy_x, rear_y + 1.8, BODY_H / 2.0 - 8.0)
        rear_dial.data.materials.append(self.m["DialKnurled"])

        # 4. Rotary Control Wheel (D-Pad)
        wheel_outer = create_cylinder("ControlWheelOuter", radius=11.0, depth=2.5, vertices=32, bevel_radius=0.4)
        wheel_outer.rotation_euler = Euler((math.radians(90.0), 0, 0))
        wheel_outer.location = (joy_x, rear_y + 1.25, -12.0)
        wheel_outer.data.materials.append(self.m["DialKnurled"])

        wheel_center = create_cylinder("ControlWheelCenterBtn", radius=4.5, depth=3.0, vertices=20, bevel_radius=0.2)
        wheel_center.rotation_euler = Euler((math.radians(90.0), 0, 0))
        wheel_center.location = (joy_x, rear_y + 1.6, -12.0)
        wheel_center.data.materials.append(self.m["AnodizedAluminum"])

        # 5. Buttons: AF-ON, AEL, Fn, Menu, Play, Trash
        for b_name, loc_z, loc_x in [
            ("Btn_AF_ON", 26.0, joy_x),
            ("Btn_AEL", 26.0, joy_x + 12.0),
            ("Btn_Fn", 3.0, joy_x),
            ("Btn_Menu", BODY_H / 2.0 - 8.0, -BODY_W * 0.38),
            ("Btn_Playback", -28.0, joy_x - 6.0),
            ("Btn_Trash", -28.0, joy_x + 6.0),
        ]:
            btn = create_cylinder(b_name, radius=2.8, depth=1.8, vertices=16, bevel_radius=0.2)
            btn.rotation_euler = Euler((math.radians(90.0), 0, 0))
            btn.location = (loc_x, rear_y + 1.0, loc_z)
            btn.data.materials.append(self.m["AnodizedAluminum"])

    def build_hardware(self):
        """Side strap lugs, port doors, and bottom tripod socket."""
        # 1. Triangular Strap Lugs (Left & Right)
        for sign, name in [(-1, "StrapLugLeft"), (1, "StrapLugRight")]:
            lug_x = sign * (BODY_W / 2.0 + 1.5)
            lug_stud = create_cylinder(name + "_Stud", radius=3.2, depth=3.5, vertices=16)
            lug_stud.rotation_euler = Euler((0, math.radians(90.0), 0))
            lug_stud.location = (lug_x, -2.0, 18.0)
            lug_stud.data.materials.append(self.m["ChromePolished"])

            lug_ring = create_cylinder(name + "_Ring", radius=5.0, depth=1.2, vertices=16)
            lug_ring.rotation_euler = Euler((0, math.radians(90.0), 0))
            lug_ring.location = (lug_x + sign * 2.0, -2.0, 18.0)
            lug_ring.data.materials.append(self.m["ChromePolished"])

        # 2. Side Port Doors (Left side: USB-C, HDMI, Audio)
        port_bay = create_beveled_box("SidePortBayCover", 1.2, 22.0, 42.0, bevel_radius=1.0, segments=2)
        port_bay.location = (BODY_W / 2.0 + 0.6, -2.0, -4.0)
        port_bay.data.materials.append(self.m["BodyMagnesium"])

        # 3. Base Plate 1/4"-20 Tripod Mount Socket
        bot_z = -BODY_H / 2.0
        tripod_collar = create_cylinder("TripodSocketCollar", radius=7.0, depth=1.2, vertices=32, bevel_radius=0.2)
        tripod_collar.location = (LENS_CX, 0.0, bot_z - 0.6)
        tripod_collar.data.materials.append(self.m["ChromePolished"])

        tripod_hole = create_cylinder("TripodSocketThreadHole", radius=3.2, depth=3.0, vertices=24)
        tripod_hole.location = (LENS_CX, 0.0, bot_z + 0.5)
        tripod_hole.data.materials.append(self.m["AnodizedAluminum"])

        # Battery / SD card door latch on bottom
        batt_door = create_beveled_box("BatteryDoorLatch", 24.0, 32.0, 1.0, bevel_radius=0.5)
        batt_door.location = (-BODY_W * 0.28, -2.0, bot_z - 0.4)
        batt_door.data.materials.append(self.m["AnodizedAluminum"])

# -------------------------------------------------------------
# Studio Lighting & Viewport Cameras
# -------------------------------------------------------------
def setup_studio_lighting():
    """Sets up softbox studio 3-point lighting and seamless cyclorama backdrop."""
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("StudioWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        # Match neutral grey studio background of reference sheet (#A8A8AC)
        bg.inputs["Color"].default_value = (0.66, 0.65, 0.68, 1.0)
        bg.inputs["Strength"].default_value = 0.95

    # Seamless Studio Cyclorama Floor
    bm_floor = bmesh.new()
    pts = [
        (-2500, -2500, -41.2),
        (2500, -2500, -41.2),
        (2500, 450, -41.2),
        (-2500, 450, -41.2),
        (2500, 950, 400.0),
        (-2500, 950, 400.0),
        (2500, 1400, 1600.0),
        (-2500, 1400, 1600.0),
    ]
    v = [bm_floor.verts.new(p) for p in pts]
    bm_floor.faces.new((v[0], v[1], v[2], v[3]))
    bm_floor.faces.new((v[3], v[2], v[4], v[5]))
    bm_floor.faces.new((v[5], v[4], v[6], v[7]))

    mesh_floor = bpy.data.meshes.new("StudioFloorMesh")
    bm_floor.to_mesh(mesh_floor)
    bm_floor.free()
    for p in mesh_floor.polygons:
        p.use_smooth = True

    floor_obj = bpy.data.objects.new("StudioFloor", mesh_floor)
    bpy.context.collection.objects.link(floor_obj)

    mat_floor = bpy.data.materials.new(name="StudioBackdropMaterial")
    mat_floor.use_nodes = True
    bsdf_f = mat_floor.node_tree.nodes.get("Principled BSDF")
    if bsdf_f:
        bsdf_f.inputs["Base Color"].default_value = (0.66, 0.65, 0.68, 1.0)
        bsdf_f.inputs["Roughness"].default_value = 0.88
        if "Specular IOR Level" in bsdf_f.inputs:
            bsdf_f.inputs["Specular IOR Level"].default_value = 0.15
    floor_obj.data.materials.append(mat_floor)

    # Key Light (Front-Top-Left)
    key_l = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_l.energy = 580.0
    key_l.size = 350.0
    key_l.color = (1.0, 0.99, 0.98)
    key_obj = bpy.data.objects.new("KeyLight", key_l)
    bpy.context.collection.objects.link(key_obj)
    key_obj.location = (-220.0, -320.0, 220.0)
    key_obj.rotation_euler = Euler((math.radians(45.0), math.radians(-25.0), math.radians(-35.0)))

    # Fill Light (Front-Right)
    fill_l = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_l.energy = 420.0
    fill_l.size = 400.0
    fill_l.color = (0.97, 0.98, 1.0)
    fill_obj = bpy.data.objects.new("FillLight", fill_l)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.location = (250.0, -280.0, 120.0)
    fill_obj.rotation_euler = Euler((math.radians(35.0), math.radians(30.0), math.radians(40.0)))

    # Top Rim Light
    rim_l = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_l.energy = 450.0
    rim_l.size = 280.0
    rim_l.color = (1.0, 1.0, 1.0)
    rim_obj = bpy.data.objects.new("RimLight", rim_l)
    bpy.context.collection.objects.link(rim_obj)
    rim_obj.location = (0.0, 120.0, 320.0)
    rim_obj.rotation_euler = Euler((math.radians(-45.0), 0, 0))

    # Soft Front Light for studio fill
    front_l = bpy.data.lights.new(name="FrontSoftLight", type='AREA')
    front_l.energy = 320.0
    front_l.size = 600.0
    front_l.color = (1.0, 1.0, 1.0)
    front_obj = bpy.data.objects.new("FrontSoftLight", front_l)
    bpy.context.collection.objects.link(front_obj)
    front_obj.location = (0.0, -380.0, 40.0)
    front_obj.rotation_euler = Euler((math.radians(85.0), 0, 0))

def setup_cameras():
    """Sets up cameras matching the 6-view reference grid."""
    cams = {}
    ORTHO_SCALE = 220.0

    target_empty = bpy.data.objects.new("CamAimTarget", None)
    bpy.context.collection.objects.link(target_empty)
    target_empty.location = (0.0, -15.0, 10.0)

    def create_cam(name, loc, rot=None, ortho=True, ortho_scale=ORTHO_SCALE, fov=52.0, track_target=None):
        cdata = bpy.data.cameras.new(name)
        cdata.sensor_fit = 'HORIZONTAL'
        if ortho:
            cdata.type = 'ORTHO'
            cdata.ortho_scale = ortho_scale
        else:
            cdata.type = 'PERSP'
            cdata.lens = fov
        cobj = bpy.data.objects.new(name, cdata)
        bpy.context.collection.objects.link(cobj)
        cobj.location = loc
        if rot is not None:
            cobj.rotation_euler = rot
        if track_target is not None:
            tt = cobj.constraints.new(type='TRACK_TO')
            tt.target = track_target
            tt.track_axis = 'TRACK_NEGATIVE_Z'
            tt.up_axis = 'UP_Y'
        cams[name] = cobj
        return cobj

    # 1. Front View (GRID 02) — Primary reference view
    create_cam("Cam_Front", (0.0, -450.0, 11.0), Euler((math.radians(90.0), 0, 0)), ortho=True, ortho_scale=220.0)

    # 2. Hero 3/4 Perspective View (GRID 01)
    create_cam("Cam_Hero", (-270.0, -360.0, 140.0), ortho=False, fov=52.0, track_target=target_empty)

    # 3. Left Profile View (GRID 03) — Lens on LEFT, Rear on RIGHT
    create_cam("Cam_Left_Profile", (450.0, -25.0, 11.0), Euler((math.radians(90.0), 0, math.radians(90.0))), ortho=True, ortho_scale=220.0)

    # 4. Rear View (GRID 04) — Rear 3/4 perspective of LCD screen
    create_cam("Cam_Back", (260.0, 360.0, 130.0), ortho=False, fov=52.0, track_target=target_empty)

    # 5. Top View (GRID 05) — Lens UP, Grip RIGHT
    create_cam("Cam_Top", (0.0, -25.0, 450.0), Euler((0, 0, math.radians(180.0))), ortho=True, ortho_scale=220.0)

    # 6. Bottom View (GRID 06) — Low-angle 3/4 view of base
    create_cam("Cam_Bottom", (-260.0, -340.0, -160.0), ortho=False, fov=50.0, track_target=target_empty)

    bpy.context.scene.camera = cams["Cam_Front"]
    return cams

def render_all_views(cams):
    """Renders all reference viewports to project renders directory."""
    scene = bpy.context.scene
    renders = {
        "Cam_Front": "render_front_view.png",
        "Cam_Hero": "render_hero_perspective.png",
        "Cam_Left_Profile": "render_left_profile.png",
        "Cam_Back": "render_rear_view.png",
        "Cam_Top": "render_top_view.png",
        "Cam_Bottom": "render_bottom_view.png",
    }

    floor_obj = bpy.data.objects.get("StudioFloor")
    for cam_name, fname in renders.items():
        cam = cams.get(cam_name)
        if cam:
            scene.camera = cam
            if floor_obj:
                floor_obj.hide_render = (cam_name == "Cam_Bottom")
            out_p = os.path.join(renders_dir, fname)
            scene.render.filepath = out_p
            print(f"[Render] Rendering {cam_name} -> {out_p}...")
            bpy.ops.render.render(write_still=True)

    # Output standard initial_render.png
    if floor_obj:
        floor_obj.hide_render = False
    scene.camera = cams["Cam_Front"]
    init_render_path = os.path.join(renders_dir, "initial_render.png")
    scene.render.filepath = init_render_path
    print(f"[Render] Rendering initial_render.png -> {init_render_path}...")
    bpy.ops.render.render(write_still=True)
    print(f"[Render] Initial render successfully saved: {init_render_path}")

def main():
    print("=" * 60)
    print(" Procedural Generator: NEXUS Mirrorless Camera (v1.0.0)")
    print("=" * 60)
    clean_scene()
    setup_render_settings()

    mat_sys = MaterialSystem(textures_dir)
    builder = CameraBuilder(mat_sys)
    builder.build_all()

    setup_studio_lighting()
    cams = setup_cameras()

    # Ensure 3D Viewport in GUI switches to Material Preview mode so user sees the textured model
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.type = 'MATERIAL'
    except Exception:
        pass

    render_all_views(cams)
    print("[Generator] Build & render complete!")

main()
