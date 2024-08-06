from typing import List

from fastapi.testclient import TestClient

from app.tests.conftest import TEST_ENDPOINT


def test_unauthorized_access(client: TestClient, input_text_to_image: List) -> None:
    payload = {"inputs": input_text_to_image}

    response = client.post(
        TEST_ENDPOINT, json=payload, headers={"Authorization": "invalid_token"}
    )

    assert response.status_code == 401
