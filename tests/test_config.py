"""Tests for the configuration system."""

import os
import tempfile

import pytest
import yaml

from halulujah.config import ExperimentConfig, load_config


def test_default_config():
    cfg = load_config()
    assert cfg.model.name_or_path == "Qwen/Qwen3-4B-Instruct"
    assert cfg.lora.r == 16
    assert cfg.sft.num_train_epochs == 50
    assert cfg.held_out_years == [2022]


def test_load_from_yaml():
    raw = {
        "model": {"name_or_path": "custom/model"},
        "held_out_years": [2020, 2021],
        "sft": {"num_train_epochs": 5},
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(raw, f)

    cfg = load_config(path)
    assert cfg.model.name_or_path == "custom/model"
    assert cfg.held_out_years == [2020, 2021]
    assert cfg.sft.num_train_epochs == 5
    # Defaults preserved for unset fields
    assert cfg.lora.r == 16

    os.unlink(path)


def test_overrides():
    cfg = load_config(overrides={"seed": 123, "ppo": {"learning_rate": 1e-4}})
    assert cfg.seed == 123
    assert cfg.ppo.learning_rate == 1e-4


def test_paths_resolve():
    cfg = load_config(overrides={"paths": {"project_root": "/tmp/test", "data_json": "data/test.json"}})
    resolved = cfg.paths.resolve("data_json")
    assert str(resolved) == "/tmp/test/data/test.json"
