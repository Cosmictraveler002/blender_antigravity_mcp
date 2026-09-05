"""
Harness Global Configuration
=============================
Default values for Blender connection, rendering, and convergence parameters.
Projects can override any of these via their project.yaml manifest.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BlenderConfig:
    """Blender socket server connection settings."""
    host: str = "localhost"
    port: int = 9876
    timeout: float = 60.0


@dataclass
class RenderConfig:
    """Default render settings."""
    resolution: Tuple[int, int] = (1920, 1080)
    view_transform: str = "Standard"
    samples: int = 128
    engine: str = "CYCLES"
    device: str = "GPU"


@dataclass
class ConvergenceConfig:
    """Iterative refinement loop convergence criteria."""
    max_iterations: int = 8
    ciede2000_threshold: float = 3.0
    ssim_threshold: float = 0.85
    geometry_score_threshold: float = 0.85


@dataclass
class HarnessConfig:
    """Master configuration container."""
    projects_dir: str = "projects"
    blender: BlenderConfig = field(default_factory=BlenderConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    @classmethod
    def from_env(cls) -> "HarnessConfig":
        """Create config with environment variable overrides."""
        cfg = cls()
        cfg.blender.host = os.getenv("BLENDER_HOST", cfg.blender.host)
        cfg.blender.port = int(os.getenv("BLENDER_PORT", str(cfg.blender.port)))
        cfg.blender.timeout = float(os.getenv("BLENDER_TIMEOUT", str(cfg.blender.timeout)))
        cfg.render.samples = int(os.getenv("RENDER_SAMPLES", str(cfg.render.samples)))
        cfg.convergence.max_iterations = int(os.getenv("MAX_ITERATIONS", str(cfg.convergence.max_iterations)))
        return cfg


# Singleton global config
DEFAULT_CONFIG = HarnessConfig.from_env()
