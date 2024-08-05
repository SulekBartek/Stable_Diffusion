from typing import Generator, List

import pytest
from fastapi.testclient import TestClient
from sd_model.util.load_config import create_and_validate_config

from app.main import app

TEST_CFG_PATH = "app/tests/config/test_config.yaml"
TEST_ENDPOINT = "http://localhost:8001/api/v1/generate"


@pytest.fixture(scope="module")
def input_text_to_image() -> List:
    cfg = create_and_validate_config(cfg_path=TEST_CFG_PATH)

    return [
        {
            "config": {
                "mode": "text_to_image",
                "prompt": cfg.prompt,
                "uncond_prompt": cfg.uncond_prompt,
                "image_path": "",
                "strength": cfg.strength,
                "do_cfg": cfg.do_cfg,
                "cfg_scale": cfg.cfg_scale,
                "num_inference_steps": cfg.num_inference_steps,
                "seed": cfg.seed,
                "device": cfg.device,
                "idle_device": cfg.idle_device,
            }
        }
    ]


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
