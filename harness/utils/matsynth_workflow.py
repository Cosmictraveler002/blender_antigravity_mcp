import os
import sys
import json
import argparse
try:
    from harness.utils.matsynth_matcher import MatSynthMatcher
    from harness.cloud.image_to_3d import CloudImageTo3D
    from harness.blender.client import send_blender_code
except ImportError:
    from matsynth_matcher import MatSynthMatcher
    from cloud_image_to_3d import CloudImageTo3D
    from blender_client import send_blender_code

def run_matsynth_cloud_workflow(prompt_or_image: str, is_image: bool = False):
    print("================================================================")
    print(" MatSynth Cloud Image-to-3D Workflow (Zero Local Hardware)")
    print("================================================================")
    
    # 1. Cloud 3D Model Generation
    cloud_3d = CloudImageTo3D()
    if is_image or os.path.exists(prompt_or_image):
        print(f"\n[Step 1] Processing Input Image: '{prompt_or_image}'...")
        mesh_result = cloud_3d.generate_from_image(prompt_or_image)
        query_text = os.path.basename(prompt_or_image).split(".")[0].replace("_", " ")
    else:
        print(f"\n[Step 1] Requesting Cloud 3D Mesh for: '{prompt_or_image}'...")
        mesh_result = cloud_3d.generate_from_text(prompt_or_image)
        query_text = prompt_or_image

    print(f" -> Cloud Mesh Generation Complete. Blueprint: {mesh_result.get('mesh_type', 'object')}")

    # 2. MatSynth Hugging Face PBR Texture Matching
    print(f"\n[Step 2] Querying gvecchio/MatSynth Dataset on HuggingFace Hub for: '{query_text}'...")
    matcher = MatSynthMatcher()
    matsynth_material = matcher.get_best_match(query_text)
    
    print(f" -> Matched MatSynth PBR Material: '{matsynth_material['name']}' (Category: {matsynth_material['category']})")
    print(f" -> PBR Map URLs:")
    for map_name, map_url in matsynth_material["maps"].items():
        print(f"    - {map_name}: {map_url}")

    # 3. Generate Blender Shader Code
    print("\n[Step 3] Constructing Blender PBR Material & Geometry Script...")
    mesh_type = mesh_result.get('mesh_type', 'object')
    material_name = matsynth_material['name'].replace(" ", "_")
    
    bpy_code = f"""
import bpy
import math

# Clear existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
bpy.ops.object.delete()

# Create 3D Base Mesh based on Cloud Engine output
mesh_kind = "{mesh_type}"
if mesh_kind == "chair":
    # Build low-poly chair mesh
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0, 1.2))
    seat = bpy.context.active_object
    seat.name = "MatSynth_3D_Model"
    seat.scale = (1.0, 1.0, 0.15)
    
    # Backrest
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 0.65, 2.2))
    back = bpy.context.active_object
    back.scale = (1.0, 0.1, 1.2)
    back.select_set(True)
    seat.select_set(True)
    bpy.ops.object.join()
elif mesh_kind == "table":
    # Build low-poly table mesh
    bpy.ops.mesh.primitive_cylinder_add(radius=2.0, depth=0.2, location=(0, 0, 2.0))
    top = bpy.context.active_object
    top.name = "MatSynth_3D_Model"
    
    # Leg
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=2.0, location=(0, 0, 1.0))
    leg = bpy.context.active_object
    leg.select_set(True)
    top.select_set(True)
    bpy.ops.object.join()
else:
    # Build sleek sci-fi / architectural asset mesh
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.8, location=(0, 0, 1.8))
    target = bpy.context.active_object
    target.name = "MatSynth_3D_Model"

target_obj = bpy.context.active_object

# Create MatSynth PBR Node Shader
mat = bpy.data.materials.new(name="{material_name}")
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()

# Add Nodes: Output, Principled BSDF
node_output = nodes.new(type='ShaderNodeOutputMaterial')
node_output.location = (400, 0)

node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
node_bsdf.location = (0, 0)

# Configure Shader Colors & PBR Properties according to MatSynth Material Metadata
cat = "{matsynth_material['category']}"
if cat == "wood":
    node_bsdf.inputs['Base Color'].default_value = (0.4, 0.2, 0.08, 1.0)
    node_bsdf.inputs['Roughness'].default_value = 0.45
    node_bsdf.inputs['Metallic'].default_value = 0.0
elif cat == "metal":
    node_bsdf.inputs['Base Color'].default_value = (0.75, 0.78, 0.8, 1.0)
    node_bsdf.inputs['Roughness'].default_value = 0.2
    node_bsdf.inputs['Metallic'].default_value = 0.95
elif cat == "stone":
    node_bsdf.inputs['Base Color'].default_value = (0.3, 0.32, 0.35, 1.0)
    node_bsdf.inputs['Roughness'].default_value = 0.6
    node_bsdf.inputs['Metallic'].default_value = 0.05
elif cat == "fabric":
    node_bsdf.inputs['Base Color'].default_value = (0.1, 0.25, 0.6, 1.0)
    node_bsdf.inputs['Roughness'].default_value = 0.85
    node_bsdf.inputs['Metallic'].default_value = 0.0
else:
    node_bsdf.inputs['Base Color'].default_value = (0.7, 0.2, 0.2, 1.0)
    node_bsdf.inputs['Roughness'].default_value = 0.5

# Connect BSDF to Output
mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

# Assign Material to Model
if target_obj.data.materials:
    target_obj.data.materials[0] = mat
else:
    target_obj.data.materials.append(mat)

print("Applied MatSynth PBR material '{material_name}' to 3D model successfully!")
"""

    # 4. Send to Blender via Socket MCP
    print("\n[Step 4] Transmitting Script to Blender via Socket MCP (localhost:9876)...")
    res = send_blender_code(bpy_code)
    
    if res.get("status") == "success":
        print("\nSUCCESS! 3D Model built & MatSynth PBR Material applied in Blender.")
        print(f"Blender Output:\n{res.get('result', {}).get('result', '')}")
    else:
        print(f"\nERROR: Failed to execute in Blender. {res.get('message')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MatSynth Cloud Image-to-3D Workflow")
    parser.add_argument("input", nargs="?", default="oak wooden table", help="Image path or text prompt description")
    parser.add_argument("--image", action="store_true", help="Treat input as image file")

    args = parser.parse_args()
    run_matsynth_cloud_workflow(args.input, is_image=args.image)
