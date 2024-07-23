from typing import Generator, List

import pytest
from fastapi.testclient import TestClient
from sd_model.util.load_config import create_and_validate_config

from app.main import app

TEST_CFG_PATH = "sd_model/config/test_config.yaml"


@pytest.fixture(scope="module")
def input_text_to_image() -> List:
    cfg = create_and_validate_config(cfg_path=TEST_CFG_PATH)
    input_image = ""

    return [
        cfg.prompt,
        cfg.uncond_prompt,
        input_image,
        cfg.strength,
        cfg.do_cfg,
        cfg.cfg_scale,
        cfg.num_inference_steps,
        cfg.seed,
        cfg.device,
        cfg.idle_device,
    ]


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
