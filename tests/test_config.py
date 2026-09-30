import glob

import pytest

from src.config import load_config

CONFIG_FILES = sorted(glob.glob("configs/*.yaml"))


def test_configs_exist():
    assert len(CONFIG_FILES) > 0


@pytest.mark.parametrize("path", CONFIG_FILES)
def test_every_config_loads(path):
    cfg = load_config(path)
    assert cfg.name
    assert cfg.lora is not None
    assert cfg.lora.r > 0
    assert all(0 <= b <= 11 for b in cfg.lora.target_blocks)
