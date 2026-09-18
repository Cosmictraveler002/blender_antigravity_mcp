bl_info = {
    "name": "3D Gen: Blender MCP Connect (Proxy)",
    "description": "Exposes Blender to Antigravity IDE (Forwards to mcp_server.addon)",
    "author": "Antigravity",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "category": "Development",
}

import bpy
from mcp_server.addon import register as _register, unregister as _unregister

def register():
    _register()

def unregister():
    _unregister()
