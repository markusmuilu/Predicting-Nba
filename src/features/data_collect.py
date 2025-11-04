import os
import json
import requests
import pandas as pd
from unidecode import unidecode
from src.utils.exception import CustomException
import sys
'''
At first it was my intention to make own folders for every team inside the season and later put them all together, 
but as I wanted to use the avg pace of the previous games of both teams, it seems easier for me to just make
one file where I make also column team
'''
base_dir = "data/train/messy"
os.makedirs(base_dir, exist_ok=True)

teams_url = "https://api.pbpstats.com/get-teams/nba"
teams = json.loads(requests.get(teams_url).text)["teams"]

seasons = ['2024-25','2023-24','2022-23']

for season in seasons:
    season_dir = os.path.join(base_dir, season)
    os.makedirs(season_dir, exist_ok=True)
    season_df = pd.DataFrame()
    print(teams)
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
           
            team_df = pd.DataFrame(logs).sort_values(by=["Date"])
            team_df["GamesPlayed"] = range(len(team_df))

            team_df["team"] = team["text"]

            season_df = pd.concat([season_df, team_df])
            
        except Exception as e:
            raise CustomException(e, sys)
            
    file_path = os.path.join(season_dir, f"{season}.csv")
    season_df.to_csv(file_path, index=False)
    print(f"Saved {season} logs.")
            

