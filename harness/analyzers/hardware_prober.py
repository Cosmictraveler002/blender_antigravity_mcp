"""
Hardware Device Prober & Multi-Tier SVBRDF Runtime Cascade
===========================================================
Detects available compute hardware (NVIDIA CUDA, AMD/Intel OpenCL APU, CPU SIMD)
and selects the optimal SVBRDF estimation tier with zero-overhead fallback.

Tiers:
  - Tier 1: Local Discrete GPU (NVIDIA CUDA + PyTorch FP16 + safetensors)
  - Tier 2: Cloud Inference REST API (HuggingFace / Replicate)
  - Tier 3: APU/CPU Optimized Classical Photometric Engine (OpenCL 2.0 / CPU SIMD)
"""

import os
import sys
from typing import Dict, Any, Optional

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class HardwareDeviceProber:
    """Probes available compute hardware and determines active SVBRDF tier."""

    TIER_1_GPU = "tier_1_local_gpu"
    TIER_2_CLOUD = "tier_2_cloud_api"
    TIER_3_APU_CPU = "tier_3_apu_cpu"

    _cached_result: Optional[Dict[str, Any]] = None

    @classmethod
    def probe(cls, weights_path: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Probe compute hardware and environment variables to determine active tier.
        Results are cached across calls unless force_refresh=True.
        """
        if cls._cached_result is not None and not force_refresh:
            return cls._cached_result

        # Check default weights location if not specified
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "svbrdf", "model_fp16.safetensors"
            )
        weights_path = os.path.abspath(weights_path)
        neural_weights_present = os.path.isfile(weights_path)

        # 1. Probe CUDA & PyTorch
        cuda_available = False
        torch_version = None
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        # 2. Probe OpenCL via OpenCV
        opencl_available = False
        opencl_device = None
        opencl_vendor = None
        opencl_version = None
        if _CV2_AVAILABLE:
            try:
                if cv2.ocl.haveOpenCL():
                    cv2.ocl.setUseOpenCL(True)
                    if cv2.ocl.useOpenCL():
                        opencl_available = True
                        dev = cv2.ocl.Device.getDefault()
                        opencl_device = dev.name() if dev else None
                        opencl_vendor = dev.vendorName() if dev else None
                        opencl_version = dev.OpenCLVersion() if dev else None
            except Exception:
                opencl_available = False

        # 3. Detect UMA (Unified Memory Architecture - APUs)
        # AMD APUs (Raven, Picasso, Renoir, Cezanne, Vega 8/11 gfx902/gfx909 etc.) or Intel Iris/UHD
        uma_memory = False
        if opencl_available and opencl_vendor:
            vendor_lower = opencl_vendor.lower()
            device_lower = (opencl_device or "").lower()
            if "advanced micro devices" in vendor_lower or "amd" in vendor_lower:
                # AMD APU indicators
                if any(x in device_lower for x in ["gfx9", "vega", "raven", "picasso", "renoir", "cezanne", "barcelo", "rembrandt", "phoenix"]):
                    uma_memory = True
                else:
                    # In Windows APU systems, integrated graphics reports as gfx902 etc.
                    uma_memory = True
            elif "intel" in vendor_lower:
                if any(x in device_lower for x in ["iris", "uhd", "hd graphics"]):
                    uma_memory = True

        # 4. Probe Cloud API tokens
        hf_token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")
        replicate_token = os.environ.get("REPLICATE_API_TOKEN")
        cloud_api_configured = bool(hf_token or replicate_token)
        cloud_provider = None
        if hf_token:
            cloud_provider = "huggingface"
        elif replicate_token:
            cloud_provider = "replicate"

        # 5. Cascading Tier Resolution
        tier_force = os.environ.get("SVBRDF_TIER_FORCE", "").strip().lower()
        active_tier: str

        if tier_force in (cls.TIER_1_GPU, cls.TIER_2_CLOUD, cls.TIER_3_APU_CPU):
            active_tier = tier_force
            tier_source = f"env_override({tier_force})"
        else:
            if cuda_available and neural_weights_present:
                active_tier = cls.TIER_1_GPU
                tier_source = "local_cuda_neural"
            elif cloud_api_configured:
                active_tier = cls.TIER_2_CLOUD
                tier_source = f"cloud_api_{cloud_provider}"
            else:
                active_tier = cls.TIER_3_APU_CPU
                tier_source = "apu_opencl" if opencl_available else "cpu_simd"

        # Construct human-readable summary
        if active_tier == cls.TIER_1_GPU:
            device_summary = "NVIDIA CUDA Tensor Cores (Local FP16)"
        elif active_tier == cls.TIER_2_CLOUD:
            device_summary = f"Cloud REST API ({cloud_provider})"
        else:
            if opencl_available:
                device_summary = f"{opencl_device} ({opencl_vendor}) [OpenCL Accelerated, UMA={uma_memory}]"
            else:
                device_summary = "Host CPU (NumPy SIMD Vectorized)"

        result = {
            "active_tier": active_tier,
            "tier_source": tier_source,
            "cuda_available": cuda_available,
            "torch_version": torch_version,
            "opencl_available": opencl_available,
            "opencl_device": opencl_device,
            "opencl_vendor": opencl_vendor,
            "opencl_version": opencl_version,
            "neural_weights_present": neural_weights_present,
            "weights_path": weights_path,
            "cloud_api_configured": cloud_api_configured,
            "cloud_provider": cloud_provider,
            "uma_memory": uma_memory,
            "device_summary": device_summary,
        }

        cls._cached_result = result
        return result
