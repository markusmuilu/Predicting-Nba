"""Serves the data the portfolio reads (markusmuilu.page), written to R2 by the player model's loop."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from predict_nba.utils.s3_client import S3Client

router = APIRouter()
FILES = {"nba-season.json", "finland.json"}


@router.get("/portfolio/{name}")
def portfolio_file(name: str):
    if name not in FILES:
        raise HTTPException(status_code=404, detail="Not found.")
    body = S3Client().download(f"portfolio/{name}")
    if body is None:
        raise HTTPException(status_code=404, detail="Not written yet.")
    return Response(content=body, media_type="application/json", headers={"Cache-Control": "public, max-age=300"})
