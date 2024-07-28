import base64
from io import BytesIO
from typing import List

from fastapi.testclient import TestClient
from PIL import Image


def test_make_prediction(client: TestClient, input_text_to_image: List) -> None:

    payload = {"inputs": input_text_to_image}

    response = client.post(
        "http://localhost:8001/api/v1/generate",
        json=payload,
    )

    assert response.status_code == 200
    prediction_data = response.json()
    encoded_image = prediction_data["image"]
    decoded_image = base64.b64decode(encoded_image)

    img = Image.open(BytesIO(decoded_image))
    assert img.size == (512, 512)
