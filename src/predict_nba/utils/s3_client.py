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

    def load_json_list(self, key: str):
        """
        Load a JSON file from S3 that should contain a list.

        Returns:
            list: parsed JSON list, or [] if the key is missing or invalid.
        """
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            raw = resp["Body"].read().decode("utf-8").strip()
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            logger.warning(f"S3 key {key} did not contain a list, returning empty list.")
            return []
        except self.s3.exceptions.NoSuchKey:
            logger.info(f"S3 key {key} not found, starting with empty list.")
            return []
        except Exception as e:
            CustomException(f"Failed to load JSON from {key}: {e}", sys)
            return []

    def save_json_list(self, data, key: str):
        """
        Save a Python list as JSON to S3.

        Overwrites any existing value at that key.
        """
        try:
            payload = json.dumps(data, indent=2).encode("utf-8")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
            )
            logger.info(f"Saved {len(data)} entries to s3://{self.bucket}/{key}")
        except Exception as e:
            CustomException(f"Failed to save JSON to {key}: {e}", sys)

    def upload_json(self, key: str, data: dict):
        """
        Upload a Python dict as JSON to S3.
        """
        try:
            payload = json.dumps(data).encode("utf-8")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ContentType="application/json"
            )
            logger.info(f"Uploaded JSON to s3://{self.bucket}/{key}")
        except Exception as e:
            raise CustomException(f"Failed to upload JSON to {key}: {e}", sys)


    def upload_bytes(self, key: str, content: bytes):
        """
        Upload raw bytes (models, CSVs, etc.).
        """
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content
            )
            logger.info(f"Uploaded bytes to s3://{self.bucket}/{key}")
        except Exception as e:
            raise CustomException(f"Failed to upload bytes to {key}: {e}", sys)


    def download_bytes(self, key: str):
        """
        Download raw bytes from S3.
        """
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            raise CustomException(f"S3 download failed for {key}: {e}", sys)
