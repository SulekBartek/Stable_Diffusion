import numpy as np
from skimage.metrics import structural_similarity as ssim
from sd_model.predict import generate
import pytest

@pytest.mark.parametrize("seeds", [
    (42, 43),
    (500, 600),
    (4534, 9602)
])
def test_diversity_with_different_seeds(config, seeds):
    """Calculate SSIM between two images generated with different seeds and assert
    that the SSIM is less than a certain threshold indicating diversity."""

    seed1, seed2 = seeds
    cfg = config
    cfg.seed = seed1
    input_params = [
        cfg.prompt,
        cfg.uncond_prompt,
        "",
        cfg.strength,
        cfg.do_cfg,
        cfg.cfg_scale,
        cfg.num_inference_steps,
        cfg.seed,
        cfg.device,
        cfg.idle_device,
    ]
    
    image1 = generate(*input_params)
    
    cfg.seed = seed2
    input_params[7] = cfg.seed
    image2 = generate(*input_params)
    
    assert isinstance(image1, np.ndarray)
    assert isinstance(image2, np.ndarray)
    assert image1.shape == image2.shape == (512, 512, 3)
    
    similarity_score = ssim(image1, image2, multichannel=True)
    print(f"SSIM between images with seeds {seed1} and {seed2}: {similarity_score}")
    
    assert similarity_score < 0.5
