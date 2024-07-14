import argparse
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import sd_model

cfg = None

# Project Directories
PACKAGE_ROOT = Path(sd_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config/base_config.yaml"


class Config(BaseModel):
    """Master config object."""

    prompt: str
    uncond_prompt: str
    image_path: str
    output_dir: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    vocab_file_path: Optional[str] = None
    merges_file_path: Optional[str] = None
    max_length: Optional[int] = None
    package_name: Optional[str] = None
    mode: Optional[str] = None
    device: str
    idle_device: str
    ckpt_path: Optional[str] = None
    seed: int
    downsampling_ratio: Optional[int] = None
    num_inference_steps: int
    num_train_steps: Optional[int] = None
    beta_start: Optional[float] = None
    beta_end: Optional[float] = None
    vae_scale: Optional[float] = None
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


def create_and_validate_config(
    parsed_config: YAML = None, cfg_path: str = None
) -> Config:
    """Run validation on config values."""

    if parsed_config is None:
        parsed_config = fetch_config_from_yaml(cfg_path)

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(**parsed_config.data)

    return _config


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Stable Diffusion Model")
    parser.add_argument(
        "--config",
        type=str,
        default="sd_model/config/base_config.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        help="see config directory for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None

    cfg = create_and_validate_config(cfg_path=args.config)

    return cfg


config = get_parser()
