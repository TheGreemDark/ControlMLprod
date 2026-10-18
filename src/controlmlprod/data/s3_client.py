import boto3
from botocore.client import BaseClient
from botocore.config import Config

from controlmlprod.data.config import (
    AWS_DEFAULT_REGION,
    S3_ACCESS_KEY,
    S3_ENDPOINT_URL,
    S3_SECRET_KEY,
    validate_s3_config,
)


def create_s3_client() -> BaseClient:
    validate_s3_config()

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=AWS_DEFAULT_REGION,
        config=Config(s3={"addressing_style": "path"}),
    )
