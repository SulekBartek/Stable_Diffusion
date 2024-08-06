import numpy as np
from skimage.metrics import structural_similarity as ssim
from sd_model.predict import generate


def test_similarity(input_for_ssim_check):
    """Test the similarity between the generated rainy image and the real rainy image."""

    sunny_image, real_rainy_image, cfg = input_for_ssim_check

    cfg.prompt = "Rainy forest mountains"
    cfg.strength = 0.1  # Low strength for small modification of original image

    input_params = [
        cfg.prompt,
        cfg.uncond_prompt,
        sunny_image,
        cfg.strength,
        cfg.do_cfg,
        cfg.cfg_scale,
        cfg.num_inference_steps,
        cfg.seed,
        cfg.device,
        cfg.idle_device,
    ]

    generated_rainy_image = generate(*input_params)

    assert isinstance(generated_rainy_image, np.ndarray)
    assert generated_rainy_image.shape == sunny_image.shape

    similarity_score = ssim(generated_rainy_image, real_rainy_image, multichannel=True)
    print(f"SSIM between modified and real image: {similarity_score}")

    assert similarity_score > 0.8
