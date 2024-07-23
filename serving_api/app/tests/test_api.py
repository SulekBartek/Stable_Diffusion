import math
from typing import List
import numpy as np
from fastapi.testclient import TestClient


def test_pass():
    assert True


def test_make_prediction(client: TestClient, test_data: List) -> None:

    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    response = client.post(
        "http://localhost:8001/api/v1/generate",
        json=payload,
    )

    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["image"] is not None
