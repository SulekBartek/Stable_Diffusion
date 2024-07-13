import pytest
from PIL import Image
from sd_model.util.load_config import create_and_validate_config

TEST_CFG_PATH = "sd_model/config/test_config.yaml"

@pytest.fixture
def input_text_to_image():
    cfg = create_and_validate_config(cfg_path=TEST_CFG_PATH)
    input_image = ""

    return [cfg.prompt, 
            cfg.uncond_prompt, 
            input_image, 
            cfg.strength, 
            cfg.do_cfg, 
            cfg.cfg_scale, 
            cfg.num_inference_steps, 
            cfg.seed, 
            cfg.device, 
            cfg.idle_device]

@pytest.fixture
def input_image_to_image():
    cfg = create_and_validate_config(cfg_path=TEST_CFG_PATH)
    input_image = Image.open(cfg.image_path)

    return [cfg.prompt, 
            cfg.uncond_prompt, 
            input_image, 
            cfg.strength, 
            cfg.do_cfg, 
            cfg.cfg_scale, 
            cfg.num_inference_steps, 
            cfg.seed, 
            cfg.device, 
            cfg.idle_device]