# import torch_fidelity
import datetime
import time

import numpy as np
from PIL import Image

from sd_model.predict import generate


def test_text_to_image(input_text_to_image):

    st = time.time()
    output_image = generate(*input_text_to_image)
    et = time.time()

    assert isinstance(output_image, np.ndarray)
    assert output_image.shape == (512, 512, 3)
    assert output_image.dtype == np.uint8

    print(f"Time taken to generate image: {et - st}")
    assert et - st < 10000

    output_image = Image.fromarray(output_image)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_image.save(f"sd_model/images/testing/output/output_{timestamp}.jpg")


# TO DO's:
# Check diversity with different seed
# Check similarity with ssim (two images sunny/rainy)
# Check FID with real image
# check performance with it/s
# check GPU/CPU performance
# check CLIP score
# check influence of higher noise
# check different resolutions
# check the model's performance against adversarial examples
