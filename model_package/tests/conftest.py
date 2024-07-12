import pytest

from PIL import Image

from sd_model.util.load_config import config


@pytest.fixture
def input_generate_data():

    prompt = "A smiling corgi dog on the floor with a big red hat on its head, cinematic, hyper realism, high detail"
    uncond_prompt = "deformed, disfigured, poorly drawn, wrong anatomy, [extra | missing | floating | disconnected] limb, blurry"
    input_image = Image.open("sd_model/images/input/corgi.jpg")
    strength = 0.8
    do_cfg = True
    cfg_scale = 7.5
    n_inference_steps = 2
    seed = 42
    device = "cpu"
    idle_device = "cpu"

    return  prompt, uncond_prompt, input_image, strength, do_cfg, cfg_scale, n_inference_steps, seed, device, idle_device