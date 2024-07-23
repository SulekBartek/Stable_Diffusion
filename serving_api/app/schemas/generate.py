from typing import List, Optional

from pydantic import BaseModel
from sd_model.util.load_config import Config as ConfigSchema


class StableDiffusionInputSchema(BaseModel):
    config: ConfigSchema


class GenerationResults(BaseModel):
    version: str
    image: Optional[str]


class GenerationInput(BaseModel):
    inputs: List[StableDiffusionInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "config": {
                            "prompt": """red sports car, (centered),
                            driving, wide angle, mountain road""",
                            "uncond_prompt": "",
                            "image_path": "",
                            "mode": "text_to_image",
                            "device": "cpu",
                            "idle_device": "cpu",
                            "seed": 42,
                            "num_inference_steps": 10,
                            "do_cfg": True,
                            "cfg_scale": 8,
                            "strength": 0.8,
                        }
                    }
                ]
            }
        }
