import threading
from typing import List

from fastapi.testclient import TestClient

from app.tests.conftest import TEST_ENDPOINT


def make_request(client: TestClient, input_text_to_image: List) -> None:
    payload = {"inputs": input_text_to_image}
    response = client.post(
        TEST_ENDPOINT,
        json=payload,
    )
    assert response.status_code == 200


def test_load(client: TestClient, input_text_to_image: List) -> None:
    threads = []
    for _ in range(10):
        thread = threading.Thread(
            target=make_request, args=(client, input_text_to_image)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
