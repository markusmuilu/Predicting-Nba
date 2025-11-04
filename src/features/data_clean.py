import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

'''
This script cleans the data:
- Keeps both sides of each game (home & away)
- Calculates rolling averages for team stats (previous games only) and ignore rows with NaN values(under 20 games before)
- Computes opponent rolling averages and stat differentials
'''

seasons = ["2024-25"]
# Windowsize defines how many previous games needed for the average, ignores the first 10 games as not enough games played to get average
windowSize = 10

for season in seasons:
    file_path = f"data/train/messy/{season}/{season}.csv"
    data = pd.read_csv(file_path)

    # Sort for chronological order
    data = data.sort_values(["team", "Date"])

    features = ["OffPoss", "DefPoss", "Pace", "Fg3Pct", "Fg2Pct", "TsPct"]

    # Keep relevant columns
    data = data[["team", "GameId", "Date", "Opponent", "Points", "GamesPlayed"] + features].copy()

    # Compute rolling averages using windowSize (previous games only, per team)
    for column in features:
        data[f"{column}_avg"] = (
            data.groupby("team")[column]
            .transform(lambda x: x.shift(1).rolling(window=windowSize, min_periods=windowSize).mean())
        )
    
    data = data.dropna()

    
    # Build opponent dataframe with opponent rolling averages
    opp_features = [f"{column}_avg" for column in features]
    opp_df = data[["team", "GameId", "Points"] + opp_features].copy()
    opp_df.columns = ["Opponent", "GameId", "Opp_Points"] + [f"Opp_{column}_avg" for column in features]

    # Merge team stats with opponent averages
    merged = data.merge(opp_df, on=["Opponent", "GameId"], how="left")

    # Compute point differential
    merged["PointDifferential"] = merged["Points"] - merged["Opp_Points"]
    merged["TeamWin"] = merged["PointDifferential"] > 0

    # Compute feature differentials
    for col in features:
        merged[f"{col}_diff"] = merged[f"{col}_avg"] - merged[f"Opp_{col}_avg"]

    # Include both pointdifferential and teamwin, as we can try out which one yeilds better results
    merged = merged[['PointDifferential', 'TeamWin', 'OffPoss_avg', 'DefPoss_avg', 'Pace_avg', 'Fg3Pct_avg', 'Fg2Pct_avg',
       'TsPct_avg', 'Opp_Points', 'Opp_OffPoss_avg', 'Opp_DefPoss_avg',
       'Opp_Pace_avg', 'Opp_Fg3Pct_avg', 'Opp_Fg2Pct_avg', 'Opp_TsPct_avg']].dropna()
    try:
        os.makedirs("data/train/clean/" +season)
    except:
        print("directory already made")
    merged.to_csv("data/train/clean/" +season + "/" +season+ ".csv")

