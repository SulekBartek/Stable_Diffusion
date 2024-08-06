import base64
import time
from io import BytesIO
from typing import List

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.tests.conftest import TEST_ENDPOINT

MIN_RESPONSE_TIME = 60


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"prompt": "A sunny day in the mountains", "strength": 0.5},
        {"prompt": "A rainy night in the city", "strength": 0.8},
    ],
)
def test_make_prediction(
    client: TestClient, input_text_to_image: List, config_overrides: dict
) -> None:

    config = input_text_to_image[0]["config"]
    for key, value in config_overrides.items():
        if key in config:
            config[key] = value

    payload = {"inputs": [input_text_to_image[0]]}

    response = client.post(
        TEST_ENDPOINT,
        json=payload,
    )

    assert response.status_code == 200
    prediction_data = response.json()
    encoded_image = prediction_data["image"]
    decoded_image = base64.b64decode(encoded_image)

    img = Image.open(BytesIO(decoded_image))
    assert img.size == (512, 512)


def test_prediction_response_time(
    client: TestClient, input_text_to_image: List
) -> None:
    payload = {"inputs": input_text_to_image}

    start_time = time.time()
    response = client.post(
        TEST_ENDPOINT,
        json=payload,
    )
    end_time = time.time()

    assert response.status_code == 200
    assert end_time - start_time < MIN_RESPONSE_TIME


def test_invalid_input(client: TestClient) -> None:
    payload = {"inputs": ["invalid_input"]}

    response = client.post(
        TEST_ENDPOINT,
        json=payload,
    )

    assert response.status_code == 422
    assert "error" in response.json()


def test_empty_input(client: TestClient) -> None:
    payload = {"inputs": []}

    response = client.post(
        TEST_ENDPOINT,
        json=payload,
    )

    assert response.status_code == 400
    assert "error" in response.json()
