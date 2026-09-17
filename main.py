"""
Unified Entry Point for Blender Reconstruction Harness and MCP Server
======================================================================
Usage:
    python main.py run <project> [--stage <stage>] [--dry-run]
    python main.py list
    python main.py info <project>
    python main.py server   (or --server / mcp, launches the Blender MCP server)
"""

import sys
import os


def main():
    """Route execution to either the Reconstruction Harness or the MCP Server."""
    if len(sys.argv) > 1 and sys.argv[1] in ("server", "mcp", "--server", "-s"):
        sys.argv.pop(1)
        from mcp_server.server import main as server_main
        server_main()
    else:
        from harness.__main__ import main as harness_main
        harness_main()


if __name__ == "__main__":
    main()
