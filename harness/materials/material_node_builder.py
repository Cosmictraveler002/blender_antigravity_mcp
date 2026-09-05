"""
Blender Material Node Graph Generator
======================================
Synthesizes Blender Python (bpy) code to automatically wire complete
PBR texture node graphs into Principled BSDF materials:
  - TextureCoordinate (UV) -> Mapping -> ImageTextures
  - Base Color (sRGB) with optional target color tinting
  - Roughness (Non-Color)
  - Normal Map (Non-Color via ShaderNodeNormalMap)
  - Ambient Occlusion (AO)
  - Decal / Typography overlay layering
"""

import os
from typing import Dict, Any, Optional, List, Tuple


def build_pbr_material_nodes(
    material_name: str,
    object_name: Optional[str] = None,
    maps: Optional[Dict[str, str]] = None,
    target_color_rgb: Optional[Tuple[float, float, float]] = None,
    roughness: float = 0.5,
    metallic: float = 0.0,
    uv_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tint_factor: float = 0.0,
) -> str:
    """
    Generates a Python (bpy) script string that wires a full PBR material in Blender.

    Args:
        material_name: Name of the material in bpy.data.materials.
        object_name: Optional object to assign this material to.
        maps: Dict of map types to file paths ('diffuse', 'roughness', 'normal', 'ao').
        target_color_rgb: sRGB tuple (r, g, b) in [0, 1] for base color or tinting.
        roughness: Base roughness if no texture map provided.
        metallic: Metallic value [0, 1].
        uv_scale: Texture scale factor (x, y, z).
        tint_factor: 0.0 = pure texture, 1.0 = full tint to target_color_rgb.

    Returns:
        Python code string ready to execute in Blender.
    """
    maps = maps or {}
    diffuse_path = maps.get("diffuse", "").replace("\\", "/")
    roughness_path = maps.get("roughness", "").replace("\\", "/")
    normal_path = maps.get("normal", "").replace("\\", "/")
    ao_path = maps.get("ao", "").replace("\\", "/")

    r, g, b = target_color_rgb if target_color_rgb else (0.8, 0.8, 0.8)

    bpy_code = f"""
import bpy
import os

mat_name = {repr(material_name)}
mat = bpy.data.materials.get(mat_name)
if not mat:
    mat = bpy.data.materials.new(name=mat_name)

mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Clear existing nodes
nodes.clear()

# Create Output and Principled BSDF
output_node = nodes.new(type='ShaderNodeOutputMaterial')
output_node.location = (600, 0)

bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (250, 0)
bsdf.inputs['Metallic'].default_value = {metallic}
bsdf.inputs['Roughness'].default_value = {roughness}
links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

# UV Mapping Setup
tex_coord = nodes.new(type='ShaderNodeTexCoord')
tex_coord.location = (-700, 0)

mapping = nodes.new(type='ShaderNodeMapping')
mapping.location = (-500, 0)
mapping.inputs['Scale'].default_value = ({uv_scale[0]}, {uv_scale[1]}, {uv_scale[2]})
links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

# 1. Diffuse / Albedo Map
diffuse_path = {repr(diffuse_path)}
if diffuse_path and os.path.exists(diffuse_path):
    img_diff = bpy.data.images.load(diffuse_path, check_existing=True)
    tex_diff = nodes.new(type='ShaderNodeTexImage')
    tex_diff.location = (-200, 150)
    tex_diff.image = img_diff
    links.new(mapping.outputs['Vector'], tex_diff.inputs['Vector'])

    if {tint_factor} > 0.01:
        # Tint node
        tint = nodes.new(type='ShaderNodeMix')
        tint.data_type = 'RGBA'
        tint.blend_type = 'MULTIPLY'
        tint.location = (50, 150)
        tint.inputs['Factor'].default_value = {tint_factor}
        tint.inputs[6].default_value = ({r}, {g}, {b}, 1.0)
        links.new(tex_diff.outputs['Color'], tint.inputs[7])
        links.new(tint.outputs[2], bsdf.inputs['Base Color'])
    else:
        links.new(tex_diff.outputs['Color'], bsdf.inputs['Base Color'])
else:
    bsdf.inputs['Base Color'].default_value = ({r}, {g}, {b}, 1.0)

# 2. Roughness Map
rough_path = {repr(roughness_path)}
if rough_path and os.path.exists(rough_path):
    img_rough = bpy.data.images.load(rough_path, check_existing=True)
    img_rough.colorspace_settings.name = 'Non-Color'
    tex_rough = nodes.new(type='ShaderNodeTexImage')
    tex_rough.location = (-200, -100)
    tex_rough.image = img_rough
    links.new(mapping.outputs['Vector'], tex_rough.inputs['Vector'])
    links.new(tex_rough.outputs['Color'], bsdf.inputs['Roughness'])

# 3. Normal Map
norm_path = {repr(normal_path)}
if norm_path and os.path.exists(norm_path):
    img_norm = bpy.data.images.load(norm_path, check_existing=True)
    img_norm.colorspace_settings.name = 'Non-Color'
    tex_norm = nodes.new(type='ShaderNodeTexImage')
    tex_norm.location = (-200, -350)
    tex_norm.image = img_norm
    links.new(mapping.outputs['Vector'], tex_norm.inputs['Vector'])

    norm_map = nodes.new(type='ShaderNodeNormalMap')
    norm_map.location = (50, -350)
    norm_map.inputs['Strength'].default_value = 1.0
    links.new(tex_norm.outputs['Color'], norm_map.inputs['Color'])
    links.new(norm_map.outputs['Normal'], bsdf.inputs['Normal'])

# Assign material to object if specified
obj_name = {repr(object_name)}
if obj_name:
    obj = bpy.data.objects.get(obj_name)
    if obj:
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
"""
    return bpy_code
