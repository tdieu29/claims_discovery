from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = []
dev_packages = ["black==21.7b0", "flake8==3.9.2", "isort==5.9.3", "pre-commit==2.15.0"]

setup(
    python_requires=">=3.7",
    install_requires=[required_packages],
    extras_require={"test": test_packages, "dev": test_packages + dev_packages},
    entry_points={"console_scripts": ["search = app.api:app"]},
)
