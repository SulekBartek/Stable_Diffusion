from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

with open(PACKAGE_ROOT / "app" / "VERSION") as version_file:
    __version__ = version_file.read().strip()
