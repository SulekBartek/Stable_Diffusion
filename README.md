<div align="center">

# Stable Diffusion Implementation

<div align="left">

From-scratch implementation of a Stable Diffusion model, divided into two main parts: the `model_package` for the model implementation and packaging, and the `serving_api` for deployment of the model via a FastAPI interface.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [License](#license)

## Project Structure

```
stable-diffusion/
│
├── model_package/
│   ├── sd_model/
│   │   ├── config/
│   │   ├── data/
│   │   ├── model/
│   │   │   ├── clip.py
│   │   │   ├── sampler.py
│   │   │   ├── unet_diffusion.py
│   │   │   └── vae.py
│   │   ├── util/
│   │   ├── finetuning.py
│   │   └── predict.py
│   ├── tests/
│   ├── requirements/
│   ├── setup.py
│   └── tox.ini
│
└── serving_api/
    ├── app/
    │   ├── api.py
    │   ├── config.py
    │   ├── main.py
    │   ├── schemas/
    │   └── tests/
    ├── Procfile
    ├── requirements.txt
    └── tox.ini
```

## Setup

## Usage

### Model Package

#### Finetuning

#### Prediction

### Serving API

Model can be deployed via CI/CD pipeline implemented in Jenkinsfile (currently its Railway deployment). If one wants to test API locally:

1. Navigate to the `serving_api` directory:

```bash
cd ../serving_api
```

2. Start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

3. The API will be accessible at `http://localhost:8001`.

#### API Endpoints

- `POST /api/v1/generate`: Generate an image from a text prompt.

Example payload:

```json
{
    "inputs": [
        {
            "config": {
                "mode": "text_to_image",
                "prompt": "A beautiful painting of a sunset over a mountain range",
                "uncond_prompt": "",
                "strength": 0.75,
                "do_cfg": True,
                "cfg_scale": 7.5,
                "num_inference_steps": 50,
                "seed": 42,
                "device": "cuda",
                "idle_device": "cpu"
            }
        }
    ]
}
```

## Configuration

Configuration files for the model are located in the `model_package/sd_model/config` directory. Adjust the `base_config.yaml` file as needed for your setup.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
