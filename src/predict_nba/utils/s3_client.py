"""
S3 helper utilities for JSON list storage used by daily automation.

Provides:
- load_json_list(key): load a JSON list from S3 (or [] if missing)
- save_json_list(data, key): save a list as JSON to S3
"""

import json
import os
import sys

import boto3
from dotenv import load_dotenv

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class S3Client:
    """Minimal JSON helper for S3."""

    def __init__(self):
        load_dotenv()
        bucket = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION")

        if not bucket:
            raise CustomException("AWS_S3_BUCKET_NAME must be set in environment.", sys)

        try:
            self.bucket = bucket
            self.s3 = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except Exception as e:
            raise CustomException(f"S3 initialization failed: {e}", sys)

    def upload(self, key: str, content: bytes, content_type=None):
        """
        Upload to s3 (models, CSVs, etc.).
        """
        try:
            if content_type is not None:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content,
                    ContentType=content_type,
                )
            else:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content
                )
            logger.info(f"Uploaded bytes to s3://{self.bucket}/{key}")
        except Exception as e:
            raise CustomException(f"Failed to upload bytes to {key}: {e}", sys)


    def download(self, key: str):
        """
        Download raw bytes from S3.
        """
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            CustomException(f"S3 download failed for {key}: {e}", sys)
            return None

