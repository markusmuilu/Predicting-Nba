"""
One-time migration script: AWS S3 (test-nba1) → Cloudflare R2 (nbaprediction)
Run from project root with both .env (AWS) and R2 credentials set.
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# ── Source: AWS S3 ──────────────────────────────────────────────────────────
aws = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
SOURCE_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")  # test-nba1

# ── Destination: Cloudflare R2 ───────────────────────────────────────────────
r2 = boto3.client(
    "s3",
    endpoint_url=os.getenv("R2_ENDPOINT"),
    region_name="auto",
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)
DEST_BUCKET = "nbaprediction"

def migrate():
    print(f"Listing all objects in s3://{SOURCE_BUCKET}...")
    paginator = aws.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=SOURCE_BUCKET)

    all_keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            all_keys.append(obj["Key"])

    print(f"Found {len(all_keys)} objects to migrate.\n")

    success, failed = [], []

    for key in all_keys:
        try:
            print(f"  Copying: {key}")
            response = aws.get_object(Bucket=SOURCE_BUCKET, Key=key)
            body = response["Body"].read()
            content_type = response.get("ContentType", "application/octet-stream")

            r2.put_object(
                Bucket=DEST_BUCKET,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
            success.append(key)
            print(f"  ✓ {key}")
        except Exception as e:
            print(f"  ✗ FAILED {key}: {e}")
            failed.append(key)

    print(f"\n{'='*50}")
    print(f"Migration complete: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed keys:")
        for k in failed:
            print(f"  - {k}")

if __name__ == "__main__":
    migrate()