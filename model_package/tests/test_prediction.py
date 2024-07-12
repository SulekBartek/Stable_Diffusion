import numpy as np
from sd_model.predict import generate

def test_output_image_to_image_format(input_generate_data):

    output_image = generate(input_generate_data)

    assert isinstance(output_image, np.ndarray)
    assert output_image.shape == (512, 512, 3)
    assert output_image.dtype == np.uint8