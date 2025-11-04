from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd

seasons = ["2024-25"]#, "2023-24", "2022-23"]
data_df = pd.DataFrame()
for season in seasons:
    season_df = pd.read_csv(f"data/train/clean/{season}/{season}.csv")
    data_df = pd.concat([data_df, season_df])

y = data_df["TeamWin"]
X = data_df[['OffPoss_avg', 'DefPoss_avg', 'Pace_avg', 'Fg3Pct_avg', 'Fg2Pct_avg',
       'TsPct_avg', 'Opp_Points', 'Opp_OffPoss_avg', 'Opp_DefPoss_avg',
       'Opp_Pace_avg', 'Opp_Fg3Pct_avg', 'Opp_Fg2Pct_avg', 'Opp_TsPct_avg']]

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(
    hidden_layer_sizes=(128, 128, 128, 128),
    alpha=0.01,
    max_iter=100000,
    early_stopping=True,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
