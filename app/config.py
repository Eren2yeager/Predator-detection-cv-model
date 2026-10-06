"""Application configuration and model constants."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Class order matches the fine-tuned RF-DETR Nano checkpoint (index -> label).
CLASS_NAMES = [
    "bear",
    "cat",
    "cattle",
    "crocodile",
    "deer",
    "dog",
    "elephant",
    "fox",
    "hyena",
    "jackal",
    "leopard",
    "lion",
    "monkey",
    "person",
    "rabbit",
    "snake",
    "tiger",
    "weasel",
    "wild_boar",
    "wolf",
]

MODEL_NAME = "RFDETRNano"

# Architecture hyper-parameters read from the checkpoint's ``model_config``.
MODEL_CONFIG = {
    "encoder": "dinov2_windowed_small",
    "out_feature_indexes": [3, 6, 9, 12],
    "dec_layers": 2,
    "two_stage": True,
    "projector_scale": ["P4"],
    "hidden_dim": 256,
    "patch_size": 16,
    "num_windows": 2,
    "sa_nheads": 8,
    "ca_nheads": 16,
    "dec_n_points": 2,
    "num_queries": 300,
    "num_select": 300,
    "postprocess_trace_alpha": 0.2,
    "bbox_reparam": True,
    "lite_refpoint_refine": True,
    "layer_norm": True,
    "amp": True,
    "num_channels": 3,
    "resolution": 384,
    "group_detr": 13,
    "gradient_checkpointing": False,
    "compile": False,
    "cuda_graphs": False,
    "fused_optimizer": True,
    "positional_encoding_size": 24,
    "ia_bce_loss": True,
    "segmentation_head": False,
    "use_grouppose_keypoints": False,
    "keypoint_cross_attn": True,
    "inter_instance_kp_attn": False,
    "grouppose_keypoint_dim_downscale": 1,
    "dual_projector": False,
    "dual_projector_kp_only": False,
    "num_keypoints_per_class": [],
    "num_decoder_registers": 0,
    "mask_downsample_ratio": 4,
    "backbone_lora": False,
    "freeze_encoder": False,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    model_path: Path = BASE_DIR / "model" / "checkpoint_best_total.pth"
    device: str = "auto"
    default_threshold: float = 0.5
    max_image_mb: float = 10.0
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    warmup: bool = True


settings = Settings()
