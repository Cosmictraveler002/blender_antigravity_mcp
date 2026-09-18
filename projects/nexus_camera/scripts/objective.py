"""Project Objective Entrypoint for nexus_camera"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from objective_camera import NexusCameraCameraObjective, main

if __name__ == "__main__":
    main()
