import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

load_dotenv(PROJECT_ROOT / ".env")

S3_ENDPOINT_URL = os.getenv(
    "S3_ENDPOINT_URL",
    "http://localhost:9000",
)
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_BUCKET = os.getenv("S3_BUCKET", "controlmlprod-data")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def validate_s3_config() -> None:
    if not S3_ACCESS_KEY or not S3_SECRET_KEY:
        raise RuntimeError(
            "S3 credentials are missing. Configure S3_ACCESS_KEY "
            "and S3_SECRET_KEY in the project .env file."
        )
