import base64
import time
from io import BytesIO
from typing import List

from fastapi.testclient import TestClient
from conftest import TEST_ENDPOINT
from PIL import Image


MIN_RESPONSE_TIME = 30


@pytest.mark.parametrize("config", [
    {"prompt": "A sunny day in the mountains", "strength": 0.5},
    {"prompt": "A rainy night in the city", "strength": 0.8},
])
def test_make_prediction(client: TestClient, input_text_to_image: List) -> None:

    payload = {"inputs": input_text_to_image}

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


def test_prediction_response_time(client: TestClient, input_text_to_image: List) -> None:
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

    assert response.status_code == 400
    assert "error" in response.json()


def test_empty_input(client: TestClient) -> None:
    payload = {"inputs": []}

    response = client.post(
        TEST_ENDPOINT,
        json=payload,
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_large_input(client: TestClient, input_text_to_image: List) -> None:
    large_input = input_text_to_image * 1000
    payload = {"inputs": large_input}

    response = client.post(
        TEST_ENDPOINT,
        json=payload,
    )

    assert response.status_code == 413
    assert "error" in response.json()
