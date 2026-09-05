"""
Harness materials package for PBR texture search, downloading, and Blender node graph synthesis.
"""

from harness.materials.pbr_engine import PBRMaterialEngine, PolyHavenClient, AmbientCGClient
from harness.materials.material_node_builder import build_pbr_material_nodes

__all__ = [
    "PBRMaterialEngine",
    "PolyHavenClient",
    "AmbientCGClient",
    "build_pbr_material_nodes",
]
