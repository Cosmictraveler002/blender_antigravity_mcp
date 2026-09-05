"""
Harness CLI Entry Point
========================
Usage:
    python -m harness run <project-name> [--stage <stage>] [--dry-run]
    python -m harness list
    python -m harness info <project-name>
"""

import sys
import os
import argparse
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from harness import __version__, __harness_name__
from harness.config import DEFAULT_CONFIG
from harness.project_loader import load_project, discover_projects, list_projects


def cmd_run(args):
    """Execute the pipeline for a project."""
    print("=" * 76)
    print(f" {__harness_name__} v{__version__}")
    print(f" Project: {args.project}")
    print(f" Stage:   {args.stage or 'all'}")
    print(f" Dry Run: {args.dry_run}")
    print("=" * 76)

    harness_root = os.getcwd()
    project = load_project(args.project, harness_root)

    print(f"\n[Config] Project '{project.name}' loaded from {project.project_dir}")
    print(f"[Config] Reference: {project.reference_image}")
    print(f"[Config] Object Type: {project.object_type}")

    # Import pipeline stages
    from harness.pipeline.stage_analyze import run_analysis
    from harness.pipeline.stage_generate import run_generation
    from harness.pipeline.stage_compare import run_comparison
    from harness.pipeline.stage_refine import run_refinement
    from harness.pipeline.stage_report import run_report

    stages = {
        "analyze": run_analysis,
        "generate": run_generation,
        "compare": run_comparison,
        "refine": run_refinement,
        "report": run_report,
    }

    stage_order = ["analyze", "generate", "compare", "refine", "report"]

    if args.stage:
        if args.stage not in stages:
            print(f"ERROR: Unknown stage '{args.stage}'. Valid: {', '.join(stage_order)}")
            sys.exit(1)
        stage_order = [args.stage]

    for stage_name in stage_order:
        print(f"\n{'=' * 76}")
        print(f" STAGE: {stage_name.upper()}")
        print(f"{'=' * 76}")

        if args.dry_run:
            print(f"  [DRY RUN] Would execute stage '{stage_name}'")
            continue

        t0 = time.time()
        try:
            stages[stage_name](project, DEFAULT_CONFIG)
            elapsed = time.time() - t0
            print(f"\n  [DONE] Stage '{stage_name}' completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  [FAIL] Stage '{stage_name}' failed after {elapsed:.1f}s: {e}")
            if not args.continue_on_error:
                sys.exit(1)

    print(f"\n{'=' * 76}")
    print(" PIPELINE COMPLETE")
    print(f"{'=' * 76}")


def cmd_list(args):
    """List all discovered projects."""
    list_projects()


def cmd_info(args):
    """Show detailed info about a project."""
    project = load_project(args.project)
    print(f"Project: {project.name}")
    print(f"Version: {project.version}")
    print(f"Type:    {project.object_type}")
    print(f"Dir:     {project.project_dir}")
    print(f"Ref:     {project.reference_image}")
    print(f"Gen:     {project.generate_script or 'N/A'}")
    print(f"Verify:  {project.verify_script or 'N/A'}")
    if project.convergence:
        c = project.convergence
        print(f"Convergence: max_iter={c.max_iterations}, dE<{c.ciede2000_threshold}, SSIM>{c.ssim_threshold}")
    if project.render:
        r = project.render
        print(f"Render: {r.resolution[0]}x{r.resolution[1]}, {r.samples}spp, {r.view_transform}")


def main():
    parser = argparse.ArgumentParser(
        prog="harness",
        description=f"{__harness_name__} v{__version__} - Multi-project 2D->3D reconstruction pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Execute pipeline for a project")
    run_parser.add_argument("project", help="Project name (directory name under projects/)")
    run_parser.add_argument("--stage", choices=["analyze", "generate", "compare", "refine", "report"],
                           help="Run only a specific stage (default: all)")
    run_parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    run_parser.add_argument("--continue-on-error", action="store_true", help="Don't stop on stage failure")
    run_parser.set_defaults(func=cmd_run)

    # list
    list_parser = subparsers.add_parser("list", help="List all discovered projects")
    list_parser.set_defaults(func=cmd_list)

    # info
    info_parser = subparsers.add_parser("info", help="Show project details")
    info_parser.add_argument("project", help="Project name")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
