import time
from predict_nba.utils.s3_client import S3Client
from predict_nba.utils.logger import logger

MODEL_KEY = "models/prediction_model.skops"
TEAMS_KEY = "teams/teams.json"

def wait_for_required_files():
    """
    Block startup until required files exist in S3.
    """
    s3 = S3Client()

    required = {
        "model": MODEL_KEY,
        "teams": TEAMS_KEY,
    }

    while True:
        ready = True

        for name, key in required.items():
            try:
                s3.s3.head_object(Bucket=s3.bucket, Key=key)
                logger.info(f"{name} OK → s3://{s3.bucket}/{key}")
            except Exception:
                logger.warning(f"{name} NOT READY → waiting for s3://{s3.bucket}/{key}")
                ready = False

        if ready:
            logger.info("All required S3 files are available. Continuing startup.")
            return

        time.sleep(5)
