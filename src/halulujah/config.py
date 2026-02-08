"""Dataclass-based configuration with YAML loading."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    project_root: str = "."
    data_json: str = "data/EGNIVIA.json"
    pdf_dir: str = "data/NVDA_10-K"
    output_dir: str = "outputs"
    cache_dir: Optional[str] = None

    def resolve(self, key: str) -> Path:
        val = getattr(self, key)
        if val is None:
            return None
        p = Path(val)
        if not p.is_absolute():
            p = Path(self.project_root) / p
        return p


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-4B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True
    device_map: Optional[str] = None
    backend: str = "transformers"  # "transformers" or "ollama"
    ollama_model: str = "qwen3:4b"


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: str = "all-linear"


@dataclass
class SFTConfig:
    num_train_epochs: int = 50
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.2
    lr_scheduler_type: str = "cosine"
    max_length: int = 2048
    packing: bool = True
    bf16: bool = True
    logging_steps: int = 20
    save_total_limit: int = 1
    seed: int = 0
    gradient_accumulation_steps: int = 1
    deepspeed: Optional[str] = None
    train_split_ratio: float = 0.9


@dataclass
class PPOConfig:
    learning_rate: float = 1.4e-5
    batch_size: int = 8
    mini_batch_size: int = 2
    num_ppo_epochs: int = 4
    kl_coef: float = 0.1
    gamma: float = 0.99
    gradient_accumulation_steps: int = 1
    # RL-specific LoRA (smaller for stability)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    max_new_tokens: int = 128
    max_steps: Optional[int] = None
    use_cot: bool = True
    use_context: bool = True


@dataclass
class OracleConfig:
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    length_penalty_weight: float = 0.05
    max_answer_words: int = 200


@dataclass
class SweepConfig:
    temperature_min: float = 0.4
    temperature_max: float = 2.0
    temperature_step: float = 0.2
    top_p_step: float = 0.2
    top_k_values: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    num_samples: int = 10
    max_new_tokens: int = 256
    use_cot: bool = True
    use_context: bool = True


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    dataset: str = "egnivia"  # "egnivia" or "templama"
    held_out_years: List[int] = field(default_factory=lambda: [2022])
    seed: int = 42


def _merge_dict(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_dataclass(cls, d: dict):
    """Construct a dataclass from a (possibly nested) dict, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return d

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for fname, ftype in field_types.items():
        if fname not in d:
            continue
        val = d[fname]
        # Check if the field type is itself a dataclass
        if dataclasses.is_dataclass(ftype):
            kwargs[fname] = _dict_to_dataclass(ftype, val) if isinstance(val, dict) else val
        else:
            kwargs[fname] = val
    return cls(**kwargs)


def load_config(
    yaml_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
) -> ExperimentConfig:
    """Load config from YAML file with optional dict overrides."""
    raw = {}
    if yaml_path is not None:
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}
    if overrides:
        _merge_dict(raw, overrides)

    # Env-var substitution for common paths
    for key in ("cache_dir", "project_root"):
        env_key = f"HALULUJAH_{key.upper()}"
        env_val = os.environ.get(env_key)
        if env_val and key not in raw.get("paths", {}):
            raw.setdefault("paths", {})[key] = env_val

    return _dict_to_dataclass(ExperimentConfig, raw)
