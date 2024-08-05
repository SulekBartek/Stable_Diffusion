import os
import pytest
import numpy as np
from PIL import Image
from sd_model.util.load_config import create_and_validate_config

TEST_CFG_PATH = "sd_model/config/test_config.yaml"
IMAGES_FOR_SSIM = "sd_model/images/ssim"
IMAGES_FOR_FID = "sd_model/images/fid"


@pytest.fixture
def config():
    return create_and_validate_config(cfg_path=TEST_CFG_PATH)


@pytest.fixture
def input_text_to_image():
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


@pytest.fixture
def input_image_to_image():
    cfg = create_and_validate_config(cfg_path=TEST_CFG_PATH)
    input_image = Image.open(cfg.image_path)

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


@pytest.fixture
def input_for_ssim_check(config):
    sunny_image = np.array(Image.open(IMAGES_FOR_SSIM + '/sunny_mountain.jpg').convert("RGB"))
    rainy_image = np.array(Image.open(IMAGES_FOR_SSIM + '/rainy_mountain.jpg').convert("RGB"))
    
    cfg = config

    return sunny_image, rainy_image, cfg


@pytest.fixture
def real_images():
    """Fixture to load real images for FID calculation"""

    images = []
    image_files_paths = os.listdir(IMAGES_FOR_FID)
    
    for image_file in image_files_paths:
        image = Image.open(os.path.join(IMAGES_FOR_FID, image_file))
        images.append(np.array(image))

    return images