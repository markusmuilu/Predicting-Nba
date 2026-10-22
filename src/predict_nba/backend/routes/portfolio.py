"""Serves the data the portfolio reads (markusmuilu.page), written to R2 by the player model's loop."""

import re

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from predict_nba.backend.limiter import limiter
from predict_nba.utils.logger import logger
from predict_nba.utils.s3_client import S3Client

router = APIRouter()
FILES = {"nba-season.json", "finland.json", "finland_predictions.jsonl"}


@router.get("/portfolio/{name}")
def portfolio_file(name: str):
    if name not in FILES:
        raise HTTPException(status_code=404, detail="Not found.")
    body = S3Client().download(f"portfolio/{name}")
    if body is None:
        raise HTTPException(status_code=404, detail="Not written yet.")
    kind = "application/x-ndjson" if name.endswith(".jsonl") else "application/json"
    return Response(content=body, media_type=kind, headers={"Cache-Control": "public, max-age=300"})


_TEAM_RE = re.compile(r"^[A-Z]{2,4}$")


@router.get("/player/predict")
@limiter.limit("10/minute")
def player_predict(request: Request, home: str = Query(...), away: str = Query(...)):
    """The player model for any two teams with today's rosters and injury list, e.g. /player/predict?home=BOS&away=NYK."""
    if not _TEAM_RE.match(home) or not _TEAM_RE.match(away) or home == away:
        raise HTTPException(status_code=400, detail="Two different team abbreviations, e.g. home=BOS&away=NYK.")
    from predict_nba.player.matchup import predict
    try:
        return predict(home, away)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"player predict {home}-{away} failed: {e}")
        raise HTTPException(status_code=503, detail="The player model is not available right now.")
