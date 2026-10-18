import hashlib
import logging
import subprocess
from pathlib import Path

from botocore.client import BaseClient
from botocore.exceptions import ClientError

from controlmlprod.data.config import (
    RAW_DIR,
    S3_BUCKET,
)
from controlmlprod.data.s3_client import create_s3_client

LOGGER = logging.getLogger(__name__)

SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
S3_KEY = "raw/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"

# MD5 опубликован на официальной странице CIFAR-10.
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def object_exists(s3: BaseClient, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if error_code in {"404", "NoSuchKey", "NotFound"} or status_code == 404:
            return False

        raise

    return True


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if calculate_md5(destination) == EXPECTED_MD5:
            LOGGER.info("Using verified local archive: %s", destination)
            return

        LOGGER.warning("Removing invalid local archive: %s", destination)
        destination.unlink()

    temporary_path = destination.with_suffix(".download")

    try:
        command = [
            "curl",
            "--fail",
            "--location",
            "--show-error",
            "--progress-bar",
            "--retry",
            "3",
            "--connect-timeout",
            "20",
            "--output",
            str(temporary_path),
        ]

        # Продолжаем загрузку с места остановки, если временный
        # файл уже существует.
        if temporary_path.exists():
            command.extend(["--continue-at", "-"])

        command.append(SOURCE_URL)

        LOGGER.info("Downloading CIFAR-10 archive from the official source")
        subprocess.run(command, check=True)

        actual_md5 = calculate_md5(temporary_path)
        if actual_md5 != EXPECTED_MD5:
            raise ValueError(
                f"CIFAR-10 archive checksum mismatch: expected {EXPECTED_MD5}, got {actual_md5}"
            )

        temporary_path.replace(destination)
        LOGGER.info("Archive downloaded and checksum verified")

    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    s3 = create_s3_client()

    if object_exists(s3, S3_BUCKET, S3_KEY):
        LOGGER.info(
            "Raw object already exists; refusing to overwrite s3://%s/%s",
            S3_BUCKET,
            S3_KEY,
        )
        return

    local_archive = RAW_DIR / ARCHIVE_NAME
    download_archive(local_archive)

    LOGGER.info("Uploading archive to s3://%s/%s", S3_BUCKET, S3_KEY)
    s3.upload_file(str(local_archive), S3_BUCKET, S3_KEY)

    LOGGER.info("Upload completed successfully")


if __name__ == "__main__":
    main()
