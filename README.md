# Blender MCP Reconstruction Harness & Multi-Project Pipeline

[![Windows](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Antigravity](https://img.shields.io/badge/Editor-Antigravity-purple?logo=visual-studio-code&logoColor=white)](https://antigravity.google)
[![Blender](https://img.shields.io/badge/Blender-4.0%2B-orange?logo=blender&logoColor=white)](https://www.blender.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)

A production-grade, multi-modal **2D Reference Image to 3D Blender Reconstruction & Self-Refining Harness** with integrated **Model Context Protocol (MCP)** server for AI-driven 3D modeling in Google Antigravity.

### Key Capabilities & Architectural Highlights
- 🔄 **Auto-Rebuild on Structural Recommendations**: Automatically escalates from micro parameter nudges to clean procedural re-generation in Blender whenever topological or aspect ratio discrepancies are detected.
- 🎨 **Packaging Diffuse Substrate Tinting (`LabelTint`)**: Inserts dynamic `ShaderNodeMix` RGBA Multiply nodes in the shader graph, preventing packaging textures from blocking closed-loop color calibration.
- 🎯 **Multi-Gate Component Convergence**: Rigorous validation enforcing aspect ratio ($\le 2\%$), component vertical span ($\le 0.8\%$), perceptual color difference ($\Delta E_{00} \le 6.5$), and radial contour MAE ($\le 0.040$) for $\ge 92.0\%$ fidelity.
- 📸 **14-Camera Multi-Viewport 360° Capture**: Automated spherical orbit verification capturing orthographic and perspective views with symmetry IoU analysis.

---

## 🌟 Architecture Overview

The codebase is split into two cleanly separated tiers:
1. **The Reusable Harness (`harness/`)**: An analysis, comparison, and iterative self-refining engine that is completely object-agnostic.
2. **Isolated Project Workspaces (`projects/<name>/`)**: Standalone project directories containing reference inputs, Blender scripts, and generated outputs.

```
Blender-MCP/
├── harness/                           # Reusable Reconstruction Engine
│   ├── __main__.py                    # CLI: python -m harness run <project>
│   ├── config.py                      # Global defaults & convergence thresholds
│   ├── project_loader.py              # Manifest parser & workspace discovery
│   ├── pipeline/                      # 6-Stage Orchestration
│   │   ├── stage_analyze.py           # Stage 1: Computer vision & feature extraction
│   │   ├── stage_generate.py          # Stage 2: Blender model synthesis
│   │   ├── stage_capture.py           # Stage Capture: Multi-viewport 360° orbit & montage
│   │   ├── stage_compare.py           # Stage 3: Render vs reference multi-modal comparison
│   │   ├── stage_refine.py            # Stage 4: Closed-loop parameter correction
│   │   └── stage_report.py            # Stage 5: Diagnostics, verdicts & summary docs
│   ├── capture/                       # Multi-Viewport 360° Capture Engine
│   │   ├── multi_viewport_renderer.py # 14-camera orbit controller & socket bridge
│   │   ├── contact_sheet_generator.py # 4x4 visual montage builder (PIL)
│   │   └── blender_scripts/           # Inside-Blender bpy camera scripts
│   │       └── multi_viewport_render.py
│   ├── analyzers/                     # Vision & Profiling Modules
│   │   ├── gemini_analyzer.py         # Multi-modal semantic & typography extraction
│   │   ├── geometry_analyzer.py       # Radial profile mesh, aspect ratios, landmarks
│   │   ├── structural_geometry_analyzer.py # 7-Step BMesh profiling, symmetry & primitives
│   │   ├── color_texture_analyzer.py  # CIE L*a*b* K-Means, Gabor filters, BSDF synthesis
│   │   ├── placement_report_generator.py # Multi-viewpoint spatial element placement
│   │   └── viewport_analyzer.py       # 360° silhouette, coverage & symmetry analyzer
│   ├── comparators/                   # Verification Engine
│   │   └── render_geometry_comparator.py # 100-level radial mesh, CIEDE2000 ΔE, Procrustes
│   ├── materials/                     # PBR Texture Retrieval & Synthesis Engine
│   │   ├── pbr_engine.py              # PolyHaven / ambientCG REST client & ranking
│   │   └── material_node_builder.py   # Procedural bpy Principled BSDF node trees
│   ├── refiners/                      # Closed-Loop Feedback Optimization Engine
│   │   ├── base_refiner.py            # Abstract controller, render triggers & convergence
│   │   ├── refiner_generator.py       # Spec-driven refiner synthesizer (zero hardcoding)
│   │   └── template_refiner.py        # LLM generation blueprint for projects
│   ├── blender/                       # Blender Communication Layer
│   │   ├── client.py                  # Socket client (port 9876)
│   │   └── addon.py                   # Blender MCP connect addon
│   └── utils/                         # Shared Mathematics & Tools
│       └── color_math.py              # Canonical sRGB ↔ XYZ ↔ Lab & CIEDE2000
│
├── projects/                          # Per-Project Workspaces
│   └── bottle/                        # Example: Water Bottle Project
│       ├── project.yaml               # Project manifest declaring scripts & tolerances
│       ├── reference/                 # Input reference imagery
│       │   └── reference_bottle.jpg
│       ├── scripts/                   # Procedural Blender construction & tuning
│       │   ├── generate_bottle_3d.py  # 3D scene construction (bpy)
│       │   ├── refine_geometry.py     # Closed-loop tuner (subclasses BaseRefinementEngine)
│       │   └── verify_and_refine_bottle.py
│       └── outputs/                   # Generated artifacts
│           ├── renders/               # Viewport & Cycles renders (plus viewports/ 14 angles)
│           ├── specs/                 # JSON design specifications
│           ├── reports/               # Markdown summaries, contact sheets & collages
│           └── textures/              # Downloaded / synthesized PBR textures
│
└── mcp_server/                        # Antigravity Model Context Protocol Server
    ├── server.py                      # Windows-native binary stdio server
    └── addon.py                       # Blender socket add-on
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Blender Add-on
1. Open Blender (4.0+ recommended).
2. Go to `Edit > Preferences > Add-ons > Install...`.
3. Select `mcp_server/addon.py` (or root `addon.py`).
4. Enable **"3D Gen: Blender MCP Connect"**.
5. Ensure the add-on server is running (defaults to `localhost:9876`).

### 3. Connect Antigravity MCP Server
Add the following to your Antigravity MCP configuration (`mcp_config.json`):
```json
{
  "mcpServers": {
    "blender": {
      "command": "C:/Path/To/Your/python.exe",
      "args": [
        "C:/Users/PC/myapps/Blender works/mcp_server/server.py"
      ],
      "env": {
        "PYTHONUTF8": "1"
      }
    }
  }
}
```

---

## ⚡ Using the Harness CLI

> [!NOTE]
> On Windows, use `py` (Python Launcher for Windows) to ensure execution with the active Python 3.12 environment. On macOS/Linux, use `python` or `python3`.

### Discover Projects
```bash
py -m harness list
```
Output:
```
Found 2 project(s):
  - bottle: Abhinav Bottle (v1.0.0) [bottle]
  - pyana_reka_can: Pyana Reka Beverage Can (v1.0.0) [can]
```

### Inspect Project Configuration
```bash
py -m harness info pyana_reka_can
```

### Execute the Full Pipeline
```bash
py -m harness run pyana_reka_can
```

### Execute a Specific Stage
```bash
# Stage 1: Computer Vision & Feature Extraction
py -m harness run pyana_reka_can --stage analyze

# Stage 2: Blender 3D Model Generation
py -m harness run pyana_reka_can --stage generate

# Stage: Multi-Viewport 360° Capture & Analysis
py -m harness run pyana_reka_can --stage capture

# Stage 3: Render vs Reference Comparison
py -m harness run pyana_reka_can --stage compare

# Stage 4: Closed-Loop Refinement Loop (with Auto-Rebuild)
py -m harness run pyana_reka_can --stage refine

# Stage 5: Final Quality Report & Metrics
py -m harness run pyana_reka_can --stage report
```

### Dry Run
```bash
py -m harness run pyana_reka_can --dry-run
```

---

## 🔬 The 6-Stage Pipeline Explained

| Stage | Name | Key Algorithms & Operations | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Analyze** | • Radial profile mesh (100 elevation slices)<br>• CIE L\*a\*b\* K-Means palette clustering<br>• Gabor filter anisotropy & frequency analysis<br>• Spatial coordinate mapping from 6 orthographic views | `geometry_design_doc.json`<br>`color_texture_design_doc.json`<br>`placement_report.json` / `.md`<br>`master_3d_design_specification.json` |
| **Stage 2** | **Generate** | • Procedural mesh synthesis in Blender<br>• Principled BSDF shader setup<br>• Procedural PBR texture baking<br>• Studio lighting & camera framing | Active 3D Blender scene<br>`initial_render.png` |
| **Stage Capture** | **Capture** | • 14-camera spherical orbit (6 ortho + 8 perspective)<br>• Transparent film background capture with headlight fill<br>• Lateral & anterior-posterior silhouette symmetry IoU<br>• 4×4 visual contact sheet montage generation | `renders/viewports/*.png`<br>`viewport_manifest.json`<br>`viewport_analysis_report.json`<br>`viewport_contact_sheet.png` |
| **Stage 3** | **Compare** | • Dual-analysis `GeometryAnalyzer` (identical model)<br>• 100-level radial mesh MAE & Procrustes metric<br>• Component CIEDE2000 color delta ($\Delta E_{00}$)<br>• Structural recommendation tagging (`is_structural`, `requires_rebuild`)<br>• 1:1 normalized baseline side-by-side diagnostic | `comparison_report.json`<br>`render_geometry_annotated.png`<br>`comparison_side_by_side.png` |
| **Stage 4** | **Refine** | • Closed-loop proportional feedback controller<br>• Auto-Rebuild escalation on structural recommendations<br>• Multi-gate component convergence validation<br>• Diffuse substrate tinting (`LabelTint` multiplier) | Updated Blender scene<br>`refinement_log.json` |
| **Stage 5** | **Report** | • Multi-metric aggregation<br>• Pass/Fail verdicts against configurable tolerances<br>• Actionable recommendations for human review | `final_report.json`<br>`final_report.md` |

---

## 📁 Creating a New Project

Creating a new reconstruction project is as simple as creating a directory under `projects/`:

1. Create directory `projects/<your_project>/` with subfolders:
   ```
   projects/<your_project>/
   ├── project.yaml
   ├── reference/
   │   └── reference.jpg
   └── scripts/
       └── generate_model.py
   ```
2. Define `project.yaml`:
   ```yaml
   name: "My Custom Model"
   version: "1.0.0"
   object_type: "custom"
   reference_image: "reference/reference.jpg"
   blender_scripts:
     generate: "scripts/generate_model.py"
   convergence:
     max_iterations: 6
     ciede2000_threshold: 4.0
     ssim_threshold: 0.85
     geometry_score_threshold: 0.85
   render:
     resolution: [1920, 1080]
     view_transform: "Standard"
     samples: 128
   viewport_capture:
     enabled: true
     resolution_scale: 0.5
     samples: 64
     padding_factor: 1.25
     camera_distance_factor: 3.2
   ```
3. Run the harness:
   ```bash
   python -m harness run <your_project>
   ```

---

## 🛠️ Testing & Quality Verification

```bash
# Verify harness project discovery
python -c "from harness.project_loader import discover_projects; print(discover_projects())"

# Verify color math canonical functions
python -c "from harness.utils.color_math import srgb_to_xyz, xyz_to_lab, compute_ciede2000; print('Color math OK')"

# Verify stage execution
python -m harness run bottle --stage analyze
```

---

## 📄 License & Credits

Personal Use & Non-Commercial License. Strictly for personal, educational, and non-monetized hobbyist use. Commercial use, resale, and monetization are strictly prohibited. See [LICENSE](file:///C:/Users/PC/myapps/Blender%20works/LICENSE) for full legal terms.
