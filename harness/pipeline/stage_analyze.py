"""
Stage 1: Multi-Modal Image Analysis & PBR Material Retrieval
=============================================================
Orchestrates the unified multi-modal vision and material pipeline:
  1. GeminiVisionAnalyzer → Semantic descriptors, PBR keywords, typography & lighting
  2. GeometryAnalyzer → Dimensions, 100-level radial mesh, silhouette polynomials
  3. ColorTextureAnalyzer → CIE Lab K-Means palette, Gabor texture isotropy, BSDF
  4. PlacementReportGenerator → 6-view spatial element placement report
  5. PBRMaterialEngine → Queries PolyHaven/ambientCG, downloads maps & procedural fallback
  6. Merges all into master_3d_design_specification.json
"""

import os
import json
from typing import Any, Dict

from harness.config import HarnessConfig
from harness.project_loader import ProjectDefinition


def run_analysis(project: ProjectDefinition, config: HarnessConfig) -> Dict[str, Any]:
    """
    Execute Stage 1: Analyze reference image and resolve PBR materials.

    Reads:
        - project.reference_image
        - Optional: projects/<project>/outputs/specs/gemini_vision_analysis.json

    Writes to project.specs_dir:
        - gemini_vision_analysis.json
        - geometry_design_doc.json
        - geometry_analysis_annotated.png
        - color_texture_design_doc.json
        - color_texture_swatches.png
        - placement_report.json
        - material_manifest.json
        - master_3d_design_specification.json
    Writes to project.reports_dir:
        - placement_report.md
    Writes to project.textures_dir:
        - Downloaded or synthesized PBR texture maps per component

    Returns:
        The master design specification dict.
    """
    from harness.analyzers.gemini_analyzer import GeminiVisionAnalyzer
    from harness.analyzers.geometry_analyzer import GeometryAnalyzer
    from harness.analyzers.color_texture_analyzer import ColorTextureAnalyzer
    from harness.analyzers.placement_report_generator import PlacementReportGenerator
    from harness.materials.pbr_engine import PBRMaterialEngine

    image_path = project.reference_image
    output_dir = project.specs_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(project.reports_dir, exist_ok=True)
    os.makedirs(project.textures_dir, exist_ok=True)

    gemini_json = os.path.join(output_dir, "gemini_vision_analysis.json")
    geom_json = os.path.join(output_dir, "geometry_design_doc.json")
    geom_vis = os.path.join(output_dir, "geometry_analysis_annotated.png")
    color_json = os.path.join(output_dir, "color_texture_design_doc.json")
    color_vis = os.path.join(output_dir, "color_texture_swatches.png")
    placement_json = os.path.join(output_dir, "placement_report.json")
    placement_md = os.path.join(project.reports_dir, "placement_report.md")
    material_manifest_json = os.path.join(output_dir, "material_manifest.json")
    master_json = os.path.join(output_dir, "master_3d_design_specification.json")

    print("=" * 76)
    print(" ADVANCED MULTI-MODAL 2D IMAGE TO 3D DESIGN ANALYSIS FRAMEWORK (v4.0.0)")
    print(f" Project: {project.name}")
    print(f" Reference Image: {image_path}")
    print("=" * 76)

    # 1. Gemini Vision Semantic Analysis
    print("\n>>> STAGE 1.1: GEMINI VISION SEMANTIC & TYPOGRAPHY PROFILING")
    gemini_engine = GeminiVisionAnalyzer()
    gemini_doc = gemini_engine.analyze(image_path=image_path, existing_spec_path=gemini_json)
    with open(gemini_json, "w", encoding="utf-8") as f:
        json.dump(gemini_doc, f, indent=2)
    print(f"[GeminiVision] Components Identified: {len(gemini_doc.get('components', []))}")
    for c in gemini_doc.get("components", []):
        print(f"  - {c.get('display_name', c['component_id'])}: {c.get('category')} ({c.get('color_hex', 'N/A')})")
    print(f"[GeminiVision] Typography & Labels: {len(gemini_doc.get('typography_and_labels', []))} found")
    for t in gemini_doc.get("typography_and_labels", []):
        print(f"  - Text '{t.get('text')}': {t.get('font_style')} on {t.get('target_component')} ({t.get('orientation')})")

    # 2. Geometry Analysis
    print("\n>>> STAGE 1.2: GEOMETRIC PROPORTIONS, RADIAL PROFILE MESH & SILHOUETTE")
    geom_engine = GeometryAnalyzer(image_path)
    geom_doc = geom_engine.analyze(output_json_path=geom_json, output_vis_path=geom_vis, gemini_spec_path=gemini_json)

    # 3. Color & Texture Analysis
    print("\n>>> STAGE 1.3: SCIENTIFIC COLOR PALETTE, CIEDE2000 & SURFACE TEXTURE PROFILING")
    color_engine = ColorTextureAnalyzer(image_path, geometry_doc_path=geom_json)
    color_doc = color_engine.analyze(output_json_path=color_json, output_vis_path=color_vis)

    # 4. Element Placement & Multi-Viewpoint Report
    print("\n>>> STAGE 1.4: ELEMENT PLACEMENT & MULTI-VIEWPOINT SPATIAL SYNTHESIS")
    placement_gen = PlacementReportGenerator(geom_json_path=geom_json, color_json_path=color_json)
    placement_doc = placement_gen.generate_report(output_json=placement_json, output_md=placement_md)

    # 5. PBR Material Retrieval & Download Engine
    print("\n>>> STAGE 1.5: PBR REPOSITORY SEARCH (POLYHAVEN/AMBIENTCG) & TEXTURE DOWNLOAD")
    pbr_engine = PBRMaterialEngine()
    materials_resolved = {}

    components = gemini_doc.get("components", [])
    if not components:
        # Fallback to color doc components if Gemini has none
        components = [{"component_id": "main_body", "category": "metal", "pbr_material_keywords": ["metal"]}]

    for comp in components:
        cid = comp["component_id"]
        cat = comp.get("category", "generic")
        keywords = comp.get("pbr_material_keywords", [cat])
        target_color = comp.get("color_hex")

        # Prioritize physical measured parameters from ColorTextureAnalyzer if available
        comp_mat = color_doc.get("component_materials", {}).get(cid, {}).get("principled_bsdf", {})
        roughness = comp_mat.get("roughness", comp.get("estimated_roughness", 0.5))
        metallic = comp_mat.get("metallic", comp.get("estimated_metallic", 0.0))

        comp_tex_dir = os.path.join(project.textures_dir, cid)
        resolved = pbr_engine.resolve_material(
            component_id=cid,
            keywords=keywords,
            category=cat,
            target_dir=comp_tex_dir,
            target_color_hex=target_color,
            estimated_roughness=roughness,
            estimated_metallic=metallic,
            resolution="1k"
        )
        materials_resolved[cid] = resolved

    with open(material_manifest_json, "w", encoding="utf-8") as f:
        json.dump(materials_resolved, f, indent=2)
    print(f"[PBREngine] Material manifest saved -> {material_manifest_json}")

    # 6. 7-Step Structural Geometry Profiling (Reference/Scene Mesh via bpy/bmesh)
    print("\n>>> STAGE 1.6: 7-STEP STRUCTURAL GEOMETRY PROFILING (BMESH)")
    structural_json = os.path.join(output_dir, "structural_geometry_report.json")
    structural_md = os.path.join(project.reports_dir, "structural_geometry_report.md")
    struct_report = {}
    try:
        from harness.analyzers.structural_geometry_analyzer import StructuralGeometryAnalyzer
        struct_analyzer = StructuralGeometryAnalyzer(num_slices=100)
        cat_hint = gemini_doc.get("overall_form", {}).get("object_type", project.object_type)
        struct_report = struct_analyzer.analyze_scene_object(
            category_hint=cat_hint,
            output_json_path=structural_json,
            output_md_path=structural_md
        )
        print(f"[StructuralGeometry] Profiling complete -> {structural_json}")
        print(f"  - Category: {struct_report.get('tier1', {}).get('dominant_category')} (Rotational Symm: {struct_report.get('tier1', {}).get('symmetry', {}).get('rotational_z', {}).get('confidence')})")
        print(f"  - Primitive Segments: {struct_report.get('tier1', {}).get('primitive_segment_count')} detected")
    except Exception as e:
        print(f"[StructuralGeometry] Scene mesh profiling deferred ({e})")

    # 7. Master Specification Merge
    master_spec = {
        "analysis_meta": {
            "source_image": os.path.basename(image_path),
            "project_name": project.name,
            "project_version": project.version,
            "framework_version": "4.0.0",
            "pipeline": [
                "GeminiVisionAnalyzer",
                "GeometryAnalyzer",
                "ColorTextureAnalyzer",
                "PlacementReportGenerator",
                "PBRMaterialEngine",
                "StructuralGeometryAnalyzer"
            ]
        },
        "gemini_vision": gemini_doc,
        "geometry": geom_doc,
        "materials_and_colors": color_doc,
        "placement_and_viewpoints": placement_doc,
        "pbr_materials": materials_resolved,
        "structural_geometry": struct_report
    }

    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(master_spec, f, indent=2)

    # 8. Post-Analysis Project Refiner Synthesis (Zero Hardcoded Objects)
    print("\n>>> STAGE 1.7: PROJECT REFINER SYNTHESIS FROM ANALYZED SCHEMA")
    from harness.refiners.refiner_generator import RefinerGenerator
    refiner_path = os.path.join(project.project_dir, "scripts", "refine_geometry.py")
    if not os.path.exists(refiner_path) or not project.refine_script:
        RefinerGenerator.generate_project_refiner(project, output_script_path=refiner_path)

    print("\n" + "=" * 76)
    print(" ANALYSIS COMPLETE — DESIGN SPECIFICATION & PBR TEXTURES GENERATED")
    print(f" 1. Gemini Vision Analysis    : {gemini_json}")
    print(f" 2. Geometry Design Doc       : {geom_json}")
    print(f" 3. Geometry Annotation Image : {geom_vis}")
    print(f" 4. Color & Texture Doc       : {color_json}")
    print(f" 5. Color Swatches Image      : {color_vis}")
    print(f" 6. Placement Report (JSON)   : {placement_json}")
    print(f" 7. Placement Report (Markdown): {placement_md}")
    print(f" 8. Structural Report (JSON)  : {structural_json}")
    print(f" 9. Structural Report (MD)    : {structural_md}")
    print(f"10. Material Manifest (JSON)  : {material_manifest_json}")
    print(f"11. Master Specification      : {master_json}")
    print(f"12. Textures Directory        : {project.textures_dir}")
    print("=" * 76)

    return master_spec
