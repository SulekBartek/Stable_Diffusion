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

    output_image = Image.fromarray(output_image)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_image.save(f"sd_model/images/testing/output/output_{timestamp}.jpg")


def test_image_to_image(input_image_to_image):

    st = time.time()
    output_image = generate(*input_image_to_image)
    et = time.time()

    assert isinstance(output_image, np.ndarray)
    assert output_image.shape == (512, 512, 3)
    assert output_image.dtype == np.uint8


def test_image_resolution(input_image_to_image):

    image = input_image_to_image[2]

    # check bigger resolution
    image = image.resize((1024, 1024))
    input_image_to_image[2] = image

    output_image = generate(*input_image_to_image)

    assert isinstance(output_image, np.ndarray)
    assert output_image.shape == (512, 512, 3)
    assert output_image.dtype == np.uint8

    # check smaller resolution
    image = image.resize((256, 256))
    input_image_to_image[2] = image

    output_image = generate(*input_image_to_image)

    assert isinstance(output_image, np.ndarray)
    assert output_image.shape == (512, 512, 3)
    assert output_image.dtype == np.uint8
    

