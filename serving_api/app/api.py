import base64
from io import BytesIO
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from PIL import Image
from sd_model import __version__ as model_version
from sd_model.predict import generate

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/generate", response_model=schemas.GenerationResults, status_code=200)
async def predict(input_data: schemas.GenerationInput) -> Any:
    """
    Generate image with stable diffusion model.
    """

    logger.info(f"Generate using inputs: {input_data.inputs}")
    try:
        cfg = input_data.inputs[0].config
        if cfg.mode == "text_to_image":
            input_image = ""
        elif cfg.mode == "image_to_image":
            input_image = Image.open(cfg.image_path)
        elif cfg.mode == "impaint":
            raise NotImplementedError("Impaint mode is not implemented yet.")

        output_image = generate(
            cfg.prompt,
            cfg.uncond_prompt,
            input_image,
            cfg.strength,
            cfg.do_cfg,
            cfg.cfg_scale,
            cfg.num_inference_steps,
            cfg.seed,
            cfg.device,
            cfg.idle_device,
        )

        pil_image = Image.fromarray(output_image)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Prediction run successfully.")

    return schemas.GenerationResults(version=__version__, image=img_str)
