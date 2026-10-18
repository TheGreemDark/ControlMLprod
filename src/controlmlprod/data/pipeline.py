import logging

from controlmlprod.data.convert import convert_dataset
from controlmlprod.data.download_raw import download_raw_archive
from controlmlprod.data.publish import publish_dataset
from controlmlprod.data.split import split_dataset
from controlmlprod.data.statistics import calculate_statistics

LOGGER = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Run the complete CIFAR-10 data preparation and publishing pipeline."""
    LOGGER.info("Step 1/5: Downloading the raw archive from S3")
    archive_path = download_raw_archive()

    LOGGER.info("Step 2/5: Converting CIFAR-10 images and annotations")
    convert_dataset(archive_path)

    LOGGER.info("Step 3/5: Creating the stratified train/validation split")
    split_dataset()

    LOGGER.info("Step 4/5: Calculating training-set RGB statistics")
    calculate_statistics()

    LOGGER.info("Step 5/5: Publishing the dataset to S3")
    publish_dataset()

    LOGGER.info("Data pipeline completed successfully")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
