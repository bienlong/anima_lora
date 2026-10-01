# LoRA module building blocks. Public API re-exported here so
# `from networks.lora_modules import LoRAModule, ...` works unchanged.

from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale
from networks.lora_modules.chimera import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
)
from networks.lora_modules.dora import DoRALoRAModule
from networks.lora_modules.hydra import HydraLoRAModule, _sigma_sinusoidal_features
from networks.lora_modules.lokr import LoKrModule
from networks.lora_modules.lora import LoRAModule
from networks.lora_modules.ortho import (
    OrthoHydraLoRAModule,
    OrthoInitLoRAModule,
    OrthoLoRAModule,
)
from networks.lora_modules.stacked_experts import StackedExpertsLoRAModule
from networks.lora_modules.step_expert import StepExpertLoRAModule

__all__ = [
    "BaseLoRAModule",
    "ChimeraHydraInferenceModule",
    "ChimeraHydraLoRAModule",
    "DoRALoRAModule",
    "HydraLoRAModule",
    "LoKrModule",
    "LoRAModule",
    "OrthoHydraLoRAModule",
    "OrthoInitLoRAModule",
    "OrthoLoRAModule",
    "StackedExpertsLoRAModule",
    "StepExpertLoRAModule",
    "_absorb_channel_scale",
    "_sigma_sinusoidal_features",
]
