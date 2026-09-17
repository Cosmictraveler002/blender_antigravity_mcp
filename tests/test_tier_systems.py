"""
Automated Verification Test Suite: Multi-Tier SVBRDF, APU Optimization & Pipeline Spec Completion
==================================================================================================
Portable and hermetic test suite designed for CI and cross-platform execution (NVIDIA, APU, Intel, CPU).

Tests:
  1. HardwareDeviceProber detection, mocked hardware matrix, and environment override
  2. SVBRDFEngine end-to-end decomposition with acceptance gates and manifest metadata
  3. PBRMaterialEngine downloaded map gate validation (recovery of degenerate maps)
  4. RenderGeometryComparator dynamic tolerance widening for snapping_confidence == 0.0 (hermetic fixture)
  5. material_node_builder Voronoi micro-bump shader injection conditioning
  6. Hann window tiled patch inference for high-resolution crops (> 512x512)
"""

import os
import sys
import shutil
import tempfile
import json
from unittest import mock
import numpy as np
import cv2
from PIL import Image

from harness.analyzers.hardware_prober import HardwareDeviceProber
from harness.analyzers.svbrdf_analyzer import SVBRDFEngine
from harness.materials.pbr_engine import PBRMaterialEngine
from harness.materials.material_node_builder import build_pbr_material_nodes
from harness.comparators.render_geometry_comparator import RenderGeometryComparator


def test_1_hardware_prober():
    print("\n--- Test 1: HardwareDeviceProber Detection & Decision Matrix ---")
    
    # 1. Live system probe: Verify schema and contract
    probe = HardwareDeviceProber.probe(force_refresh=True)
    print(f"Live active tier: {probe['active_tier']}")
    print(f"Device summary:   {probe['device_summary']}")
    print(f"OpenCL available: {probe['opencl_available']} (Device: {probe['opencl_device']})")
    print(f"CUDA available:   {probe['cuda_available']}")
    print(f"UMA memory:       {probe['uma_memory']}")

    valid_tiers = [
        HardwareDeviceProber.TIER_1_GPU,
        HardwareDeviceProber.TIER_2_CLOUD,
        HardwareDeviceProber.TIER_3_APU_CPU,
    ]
    assert probe["active_tier"] in valid_tiers, f"Unknown tier {probe['active_tier']}"
    assert isinstance(probe["cuda_available"], bool)
    assert isinstance(probe["opencl_available"], bool)
    assert isinstance(probe["uma_memory"], bool)
    assert isinstance(probe["device_summary"], str)

    # 2. Test environment variable override
    for target_tier in valid_tiers:
        os.environ["SVBRDF_TIER_FORCE"] = target_tier
        override_probe = HardwareDeviceProber.probe(force_refresh=True)
        assert override_probe["active_tier"] == target_tier, f"Failed override to {target_tier}"
    del os.environ["SVBRDF_TIER_FORCE"]

    # 3. Test hardware decision matrix via hermetic mocks
    mock_torch_cuda = mock.MagicMock()
    mock_torch_cuda.cuda.is_available.return_value = True
    mock_torch_cuda.__version__ = "2.2.0"

    mock_torch_no_cuda = mock.MagicMock()
    mock_torch_no_cuda.cuda.is_available.return_value = False
    mock_torch_no_cuda.__version__ = "2.2.0"

    # Scenario A: CUDA available and neural weights present -> Tier 1
    with mock.patch.dict(sys.modules, {"torch": mock_torch_cuda}), \
         mock.patch("os.path.isfile", return_value=True):
        p = HardwareDeviceProber.probe(weights_path="/dummy/weights.safetensors", force_refresh=True)
        assert p["active_tier"] == HardwareDeviceProber.TIER_1_GPU
        assert p["cuda_available"] is True

    # Scenario B: No CUDA, but OpenCL available (APU / Intel / AMD) -> Tier 3
    with mock.patch.dict(sys.modules, {"torch": mock_torch_no_cuda}), \
         mock.patch("cv2.ocl.haveOpenCL", return_value=True), \
         mock.patch("cv2.ocl.useOpenCL", return_value=True):
        p = HardwareDeviceProber.probe(force_refresh=True)
        assert p["active_tier"] == HardwareDeviceProber.TIER_3_APU_CPU
        assert p["opencl_available"] is True

    # Scenario C: Pure CPU (no CUDA, no OpenCL), HuggingFace token provided -> Tier 2
    with mock.patch.dict(sys.modules, {"torch": mock_torch_no_cuda}), \
         mock.patch("cv2.ocl.haveOpenCL", return_value=False), \
         mock.patch.dict(os.environ, {"HUGGINGFACE_API_TOKEN": "hf_dummy_token"}, clear=True):
        p = HardwareDeviceProber.probe(force_refresh=True)
        assert p["active_tier"] == HardwareDeviceProber.TIER_2_CLOUD

    # Scenario D: Pure CPU (no CUDA, no OpenCL), no cloud token -> Tier 3 CPU fallback
    with mock.patch.dict(sys.modules, {"torch": mock_torch_no_cuda}), \
         mock.patch("cv2.ocl.haveOpenCL", return_value=False), \
         mock.patch.dict(os.environ, {}, clear=True):
        p = HardwareDeviceProber.probe(force_refresh=True)
        assert p["active_tier"] == HardwareDeviceProber.TIER_3_APU_CPU
        assert p["opencl_available"] is False

    # Restore live probe
    HardwareDeviceProber.probe(force_refresh=True)
    print("Test 1 PASSED: Hardware prober contract, overrides, and mock matrix verified.")


def test_2_svbrdf_end_to_end():
    print("\n--- Test 2: SVBRDF End-to-End Decomposition ---")
    engine = SVBRDFEngine()
    temp_dir = tempfile.mkdtemp(prefix="test_svbrdf_")

    try:
        # Create a test crop with gradient and color
        h, w = 512, 512
        y, x = np.mgrid[0:h, 0:w]
        synthetic_crop = np.zeros((h, w, 3), dtype=np.uint8)
        synthetic_crop[:, :, 0] = np.clip(180 + 40 * np.sin(x / 30.0), 0, 255).astype(np.uint8)
        synthetic_crop[:, :, 1] = np.clip(120 + 30 * np.cos(y / 25.0), 0, 255).astype(np.uint8)
        synthetic_crop[:, :, 2] = np.clip(60 + 20 * np.sin((x + y) / 40.0), 0, 255).astype(np.uint8)

        manifest = engine.decompose_crop(
            crop_bgr=synthetic_crop,
            output_dir=temp_dir,
            component_id="portable_body_test",
            base_roughness_hint=0.35,
            base_metallic_hint=0.80,
        )

        # Assert map existence and dimensions
        for map_key in ["diffuse", "roughness", "normal", "metallic"]:
            path = manifest["maps"][map_key]
            assert os.path.isfile(path), f"Missing output map: {path}"
            img = Image.open(path)
            assert img.size == (512, 512), f"Expected 512x512, got {img.size}"

        # Assert manifest contract (portable across any active tier)
        valid_tiers = ["tier_1_local_gpu", "tier_2_cloud_api", "tier_3_apu_cpu"]
        assert manifest["execution_tier"] in valid_tiers, f"Unexpected tier {manifest['execution_tier']}"
        assert isinstance(manifest["opencl_accelerated"], bool)
        assert "inference_time_ms" in manifest
        assert manifest["normal_mode"] in ["valid_tangent", "photometric_scharr_fallback"]
        assert manifest["composite_confidence_q"] > 0.0

        print(f"Manifest output: Tier={manifest['execution_tier']}, Time={manifest['inference_time_ms']}ms, OpenCL={manifest['opencl_accelerated']}, NormalMode={manifest['normal_mode']}, Q={manifest['composite_confidence_q']}")
        print("Test 2 PASSED: SVBRDF decomposition produced valid 4-map bundle with metadata.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_3_polyhaven_map_validation():
    print("\n--- Test 3: PolyHaven Downloaded Map Gate Validation ---")
    pe = PBRMaterialEngine()
    temp_dir = tempfile.mkdtemp(prefix="test_polyhaven_")

    try:
        # Synthesize deliberately degenerate downloaded maps:
        # 1. Perfectly flat normal map in RGB is [128, 128, 255] -> in BGR is [255, 128, 128] (Phi(N) = 1.0 > 0.98)
        flat_norm = np.full((256, 256, 3), [255, 128, 128], dtype=np.uint8)
        norm_path = os.path.join(temp_dir, "normal.png")
        cv2.imwrite(norm_path, flat_norm)

        # 2. Perfectly uniform roughness map (Var(R) = 0.0 < 0.005)
        flat_rough = np.full((256, 256), 128, dtype=np.uint8)
        rough_path = os.path.join(temp_dir, "roughness.png")
        cv2.imwrite(rough_path, flat_rough)

        # 3. Diffuse map with high-contrast pattern to allow Scharr normal recovery
        diff_arr = (np.random.rand(256, 256, 3) * 200 + 20).astype(np.uint8)
        diff_path = os.path.join(temp_dir, "diffuse.png")
        cv2.imwrite(diff_path, diff_arr)

        downloaded_maps = {
            "diffuse": diff_path,
            "roughness": rough_path,
            "normal": norm_path,
        }

        # Run validation
        validated = pe._validate_downloaded_maps(
            maps=downloaded_maps,
            target_dir=temp_dir,
            category="metal",
        )

        # Check recovered roughness variance
        rec_rough = cv2.imread(validated["roughness"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        rec_var = float(np.var(rec_rough))
        print(f"Recovered roughness variance: {rec_var:.5f} (Gate >= 0.005)")
        assert rec_var >= 0.005, f"Roughness recovery failed, variance {rec_var} < 0.005"

        # Check recovered normal flatness
        rec_norm_bgr = cv2.imread(validated["normal"])
        rec_norm_rgb = cv2.cvtColor(rec_norm_bgr, cv2.COLOR_BGR2RGB)
        rec_norm_float = (rec_norm_rgb.astype(np.float32) / 127.5) - 1.0
        flat_diff = np.sqrt(rec_norm_float[:, :, 0]**2 + rec_norm_float[:, :, 1]**2 + (rec_norm_float[:, :, 2] - 1.0)**2)
        flat_fraction = float(np.mean(flat_diff < 0.02))
        print(f"Recovered normal flatness: {flat_fraction:.4f} (Gate <= 0.98)")
        assert flat_fraction <= 0.98, f"Normal recovery failed, flatness {flat_fraction} > 0.98"

        print("Test 3 PASSED: Degenerate downloaded maps were recovered via acceptance gates.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_4_dynamic_tolerance_widening():
    print("\n--- Test 4: Dynamic Tolerance Widening (snapping_confidence == 0.0) ---")
    temp_dir = tempfile.mkdtemp(prefix="test_tolerance_")

    try:
        # Create a self-contained dummy render image and minimal design documents
        dummy_render_path = os.path.join(temp_dir, "dummy_render.png")
        cv2.imwrite(dummy_render_path, np.full((128, 128, 3), 128, dtype=np.uint8))

        dummy_geom_path = os.path.join(temp_dir, "dummy_geom.json")
        target_geom = {
            "overall_dimensions": {"aspect_ratio_height_to_width": 2.0},
            "components": {
                "part_unverified": {
                    "display_name": "Unverified Joint",
                    "height_ratio_to_total": 0.20,
                    "snapping_confidence": 0.0,  # Unverified prior
                },
                "part_verified": {
                    "display_name": "Verified Joint",
                    "height_ratio_to_total": 0.20,
                    "snapping_confidence": 1.0,  # Physical boundary verified
                }
            },
            "radial_profile_mesh": []
        }
        with open(dummy_geom_path, "w", encoding="utf-8") as f:
            json.dump(target_geom, f)

        dummy_color_path = os.path.join(temp_dir, "dummy_color.json")
        with open(dummy_color_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        # Initialize comparator with hermetic fixtures
        comparator = RenderGeometryComparator(
            render_image_path=dummy_render_path,
            target_geom_json=dummy_geom_path,
            target_color_json=dummy_color_path,
        )

        # Rendered result has 10% relative error (0.22 vs 0.20 -> h_err = 0.10)
        # Standard threshold: 0.05 -> verified component exceeds it and flags recommendation
        # Widened threshold: 0.05 * 2.5 = 0.125 -> unverified component passes and is absorbed
        rendered_mock = {
            "aspect_ratio": 2.0,
            "components": {
                "part_unverified": {"height_ratio_to_total": 0.22},
                "part_verified": {"height_ratio_to_total": 0.22},
            },
            "radial_profile_mesh": []
        }

        results = comparator.compare_against_target(rendered_mock)
        recs = [r["parameter"] for r in results.get("correction_recommendations", [])]
        print(f"Correction recommendations: {recs}")

        assert "part_verified_height_ratio" in recs, "Verified part with 10% error should trigger recommendation"
        assert "part_unverified_height_ratio" not in recs, "Unverified part should be absorbed by 2.5x widened tolerance"
        print("Test 4 PASSED: Dynamic tolerance widening correctly absorbs unverified boundary errors.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_5_voronoi_micro_bump_shader_injection():
    print("\n--- Test 5: Voronoi Micro-Bump Shader Graph Injection ---")
    # 1. With fallback normal mode -> MUST inject Voronoi and Bump nodes
    code_fallback = build_pbr_material_nodes(
        material_name="TestMat_Fallback",
        normal_mode="photometric_scharr_fallback"
    )
    assert "ShaderNodeTexVoronoi" in code_fallback, "Missing ShaderNodeTexVoronoi in fallback mode"
    assert "ShaderNodeBump" in code_fallback, "Missing ShaderNodeBump in fallback mode"
    assert "ShaderNodeVectorMath" in code_fallback, "Missing ShaderNodeVectorMath in fallback mode"
    assert "Scale'].default_value = 250.0" in code_fallback, "Missing Scale 250.0 parameter"
    assert "Strength'].default_value = 0.08" in code_fallback, "Missing Strength 0.08 parameter"

    # 2. With valid tangent normal mode -> MUST NOT inject Voronoi node
    code_valid = build_pbr_material_nodes(
        material_name="TestMat_Valid",
        normal_mode="valid_tangent"
    )
    assert "ShaderNodeTexVoronoi" not in code_valid, "ShaderNodeTexVoronoi should NOT be injected in valid_tangent mode"
    assert "ShaderNodeBump" not in code_valid, "ShaderNodeBump should NOT be injected in valid_tangent mode"

    print("Test 5 PASSED: Voronoi micro-bump shader injection correctly conditioned on normal_mode.")


def test_6_hann_window_tiling():
    print("\n--- Test 6: Hann Window Tiled Patch Inference (> 512x512) ---")
    engine = SVBRDFEngine()
    # Create 768x768 image to trigger tiled inference
    h, w = 768, 768
    y, x = np.mgrid[0:h, 0:w]
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = np.clip(150 + 50 * np.sin(x / 40.0), 0, 255).astype(np.uint8)
    rgb_img[:, :, 1] = np.clip(130 + 40 * np.cos(y / 35.0), 0, 255).astype(np.uint8)
    rgb_img[:, :, 2] = np.clip(90 + 30 * np.sin((x + y) / 50.0), 0, 255).astype(np.uint8)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    albedo, roughness, normal, metallic, mode = engine._extract_tier_3_apu_cpu(
        rgb_img, gray_img, target_color_hex="#96825a", roughness_hint=0.5, metallic_hint=0.0
    )

    assert mode == "photometric_intrinsic_decomposition_tiled", f"Expected tiled mode, got {mode}"
    assert albedo.shape == (768, 768, 3)
    assert roughness.shape == (768, 768)
    assert normal.shape == (768, 768, 3)
    assert metallic.shape == (768, 768)

    # Check for NaN / Inf
    assert not np.isnan(albedo).any()
    assert not np.isnan(roughness).any()
    assert not np.isnan(normal).any()

    # Check boundary continuity: examine horizontal slice across tile boundary at x=256
    # Gradient in roughness across overlap zone should be smooth (no sharp discontinuity)
    boundary_diff = np.abs(roughness[:, 255] - roughness[:, 257])
    max_boundary_step = float(np.max(boundary_diff))
    print(f"Max pixel delta across patch boundary: {max_boundary_step:.4f}")
    assert max_boundary_step < 0.15, f"Boundary step {max_boundary_step} too sharp, Hann blending failed"

    print("Test 6 PASSED: Hann window tiled decomposition successfully computed seamless high-res maps.")


if __name__ == "__main__":
    test_1_hardware_prober()
    test_2_svbrdf_end_to_end()
    test_3_polyhaven_map_validation()
    test_4_dynamic_tolerance_widening()
    test_5_voronoi_micro_bump_shader_injection()
    test_6_hann_window_tiling()
    print("\n============================================================")
    print("ALL 6 AUTOMATED VERIFICATION TESTS PASSED SUCCESSFULLY! [PASS]")
    print("============================================================\n")
