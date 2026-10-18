import os

import boto3
from botocore.config import Config


def main() -> None:
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"}),
    )

    bucket = os.getenv("S3_BUCKET", "controlmlprod-data")
    existing_buckets = {item["Name"] for item in client.list_buckets().get("Buckets", [])}

    if bucket in existing_buckets:
        print(f"Bucket already exists: {bucket}")
        return

    client.create_bucket(Bucket=bucket)
    print(f"Created bucket: {bucket}")


if __name__ == "__main__":
    main()
