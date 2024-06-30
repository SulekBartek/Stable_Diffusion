import logging

from .util.load_config import PACKAGE_ROOT, config



logging.getLogger(config.package_name).addHandler(logging.NullHandler())

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
