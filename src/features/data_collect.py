import os
import json
import requests
import pandas as pd
from unidecode import unidecode
from src.utils.exception import CustomException
import sys

base_dir = "data/train/messy"
os.makedirs(base_dir, exist_ok=True)

teams_url = "https://api.pbpstats.com/get-teams/nba"
teams = json.loads(requests.get(teams_url).text)["teams"]

seasons = ['2024-25','2023-24','2022-23']

for season in seasons:
    season_dir = os.path.join(base_dir, season)
    os.makedirs(season_dir, exist_ok=True)
    
    for team in teams:
        try:
            print(team)
            response = requests.get(
                "https://api.pbpstats.com/get-game-logs/nba",
                params={
                    "Season": season,
                    "SeasonType": "Regular Season",
                    "EntityType": "Team",
                    "EntityId": team["id"]
                }
            )
            data = response.json()
            logs = data.get("multi_row_table_data", [])
            if not logs:
                continue
            
            team_df = pd.DataFrame(logs)
            team_name = team["text"]
            file_path = os.path.join(season_dir, f"{team_name}.csv")
            team_df.to_csv(file_path, index=False)
            print(f"Saved {team_name} {season} logs.")
            
        except Exception as e:
            raise CustomException(e, sys)
