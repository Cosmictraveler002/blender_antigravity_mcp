"""
Project Loader
===============
Discovers and validates projects under the projects/ directory.
Each project must contain a project.yaml manifest.
"""

import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from harness.config import DEFAULT_CONFIG, HarnessConfig, ConvergenceConfig, RenderConfig


@dataclass
class ProjectDefinition:
    """Validated project definition from project.yaml."""
    name: str
    version: str
    object_type: str
    project_dir: str  # Absolute path to project root
    reference_image: str  # Absolute path to reference image

    # Script paths (absolute)
    generate_script: Optional[str] = None
    verify_script: Optional[str] = None
    refine_script: Optional[str] = None

    # Override-able config
    convergence: Optional[ConvergenceConfig] = None
    render: Optional[RenderConfig] = None

    # Output directories (absolute)
    outputs_dir: str = ""
    renders_dir: str = ""
    specs_dir: str = ""
    reports_dir: str = ""
    textures_dir: str = ""

    # Raw YAML content
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.outputs_dir = os.path.join(self.project_dir, "outputs")
        self.renders_dir = os.path.join(self.outputs_dir, "renders")
        self.specs_dir = os.path.join(self.outputs_dir, "specs")
        self.reports_dir = os.path.join(self.outputs_dir, "reports")
        self.textures_dir = os.path.join(self.outputs_dir, "textures")


def _resolve_path(project_dir: str, relative_path: str) -> str:
    """Resolve a path relative to the project directory."""
    return os.path.normpath(os.path.join(project_dir, relative_path))


def load_project(project_name: str, harness_root: Optional[str] = None) -> ProjectDefinition:
    """
    Load and validate a single project by name.

    Args:
        project_name: Name of the project directory under projects/.
        harness_root: Root directory of the harness (defaults to cwd).

    Returns:
        Validated ProjectDefinition instance.

    Raises:
        FileNotFoundError: If project directory or manifest doesn't exist.
        ValueError: If required fields are missing from project.yaml.
    """
    if harness_root is None:
        harness_root = os.getcwd()

    project_dir = os.path.join(harness_root, DEFAULT_CONFIG.projects_dir, project_name)
    manifest_path = os.path.join(project_dir, "project.yaml")

    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Project manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["name", "object_type", "reference_image"]
    for field_name in required_fields:
        if field_name not in cfg:
            raise ValueError(f"project.yaml missing required field: '{field_name}'")

    # Resolve reference image
    ref_image = _resolve_path(project_dir, cfg["reference_image"])
    if not os.path.isfile(ref_image):
        raise FileNotFoundError(f"Reference image not found: {ref_image}")

    # Resolve scripts
    scripts = cfg.get("blender_scripts", {})
    gen_script = None
    if "generate" in scripts:
        gen_script = _resolve_path(project_dir, scripts["generate"])
    verify_script = None
    if "verify" in scripts:
        verify_script = _resolve_path(project_dir, scripts["verify"])
    refine_script = None
    if "refine" in scripts:
        refine_script = _resolve_path(project_dir, scripts["refine"])
    elif verify_script:
        refine_script = verify_script

    # Parse convergence overrides
    convergence = None
    if "convergence" in cfg:
        conv_cfg = cfg["convergence"]
        convergence = ConvergenceConfig(
            max_iterations=conv_cfg.get("max_iterations", DEFAULT_CONFIG.convergence.max_iterations),
            ciede2000_threshold=conv_cfg.get("ciede2000_threshold", DEFAULT_CONFIG.convergence.ciede2000_threshold),
            ssim_threshold=conv_cfg.get("ssim_threshold", DEFAULT_CONFIG.convergence.ssim_threshold),
            geometry_score_threshold=conv_cfg.get("geometry_score_threshold", DEFAULT_CONFIG.convergence.geometry_score_threshold),
        )

    # Parse render overrides
    render = None
    if "render" in cfg:
        r_cfg = cfg["render"]
        res = r_cfg.get("resolution", list(DEFAULT_CONFIG.render.resolution))
        render = RenderConfig(
            resolution=tuple(res),
            view_transform=r_cfg.get("view_transform", DEFAULT_CONFIG.render.view_transform),
            samples=r_cfg.get("samples", DEFAULT_CONFIG.render.samples),
        )

    project = ProjectDefinition(
        name=cfg["name"],
        version=cfg.get("version", "1.0.0"),
        object_type=cfg["object_type"],
        project_dir=project_dir,
        reference_image=ref_image,
        generate_script=gen_script,
        verify_script=verify_script,
        refine_script=refine_script,
        convergence=convergence,
        render=render,
        raw_config=cfg,
    )

    # Ensure output directories exist
    for d in [project.outputs_dir, project.renders_dir, project.specs_dir,
              project.reports_dir, project.textures_dir]:
        os.makedirs(d, exist_ok=True)

    return project


def discover_projects(harness_root: Optional[str] = None) -> List[str]:
    """
    Discover all valid project names in the projects/ directory.

    Returns:
        List of project directory names that contain a project.yaml.
    """
    if harness_root is None:
        harness_root = os.getcwd()

    projects_dir = os.path.join(harness_root, DEFAULT_CONFIG.projects_dir)
    if not os.path.isdir(projects_dir):
        return []

    found = []
    for entry in sorted(os.listdir(projects_dir)):
        full_path = os.path.join(projects_dir, entry)
        if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "project.yaml")):
            found.append(entry)
    return found


def list_projects(harness_root: Optional[str] = None) -> None:
    """Print a formatted list of discovered projects."""
    projects = discover_projects(harness_root)
    if not projects:
        print("No projects found in projects/ directory.")
        return

    print(f"Found {len(projects)} project(s):")
    for name in projects:
        try:
            proj = load_project(name, harness_root)
            print(f"  - {name}: {proj.name} (v{proj.version}) [{proj.object_type}]")
        except Exception as e:
            print(f"  - {name}: ERROR - {e}")
