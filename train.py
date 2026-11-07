import logging
import sys

from src.pipelines.training_pipeline import (
    run_training_pipeline
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():

    logger.info(
        "Starting Customer Churn Training Pipeline..."
    )

    run_training_pipeline()

    logger.info(
        "Training Pipeline Completed Successfully."
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        logger.exception(
            f"Pipeline Failed: {e}"
        )

        raise