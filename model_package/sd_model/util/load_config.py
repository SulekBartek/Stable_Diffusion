from pathlib import Path
from typing import  Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import sd_model

# Project Directories
PACKAGE_ROOT = Path(sd_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config/base_config.yml"

class Config(BaseModel):
    """Master config object."""

    prompt: str
    uncond_prompt: str
    image_path: str
    output_dir: str
    width: int
    height: int
    vocab_file_path: str
    merges_file_path: str
    max_length: int
    package_name: str
    mode: str
    device: str
    idle_device: str
    ckpt_path: str
    seed: int
    downsampling_ratio: int
    num_inference_steps: int
    num_train_steps: int
    beta_start: float
    beta_end: float
    vae_scale: float
    do_cfg: bool
    cfg_scale: int
    strength: float

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        if CONFIG_FILE_PATH.is_file():
            cfg_path = CONFIG_FILE_PATH
        else:
            raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(**parsed_config.data)

    return _config


config = create_and_validate_config()
