import os
import sys
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
import skops.io as sio
from supabase import create_client
from dotenv import load_dotenv
from src.utils.exception import CustomException


class ModelTrainer:
    """
    Trains and stores a general-purpose prediction model (currently Logistic Regression)
    using advanced NBA statistics. Supports future flexibility for other model types.
    """

    def __init__(
        self,
        model_type="logistic_regression",
        C=1.0,
        max_iter=1000,
        test_size=0.2,
        random_state=42,
        bucket_name="modelData",
    ):
        self.model_type = model_type
        self.C = C
        self.max_iter = max_iter
        self.test_size = test_size
        self.random_state = random_state
        self.bucket_name = bucket_name

        # Full feature set — advanced, opponent, differential, and home-court stats
        self.features = [
            # Base stats
            "OffPoss_avg", "DefPoss_avg", "Pace_avg", "Fg3Pct_avg", "Fg2Pct_avg", "TsPct_avg",

            # Advanced team metrics
            "OffRtg_avg", "DefRtg_avg", "NetRtg_avg", "EfgDiff_avg", "TsDiff_avg",
            "Rebounds_avg", "Steals_avg", "Blocks_avg",

            # Opponent metrics
            "Opp_OffPoss_avg", "Opp_DefPoss_avg", "Opp_Pace_avg", "Opp_Fg3Pct_avg", "Opp_Fg2Pct_avg", "Opp_TsPct_avg",
            "Opp_OffRtg_avg", "Opp_DefRtg_avg", "Opp_NetRtg_avg", "Opp_EfgDiff_avg", "Opp_TsDiff_avg",
            "Opp_Rebounds_avg", "Opp_Steals_avg", "Opp_Blocks_avg",

            # Differentials
            "OffPoss_diff", "DefPoss_diff", "Pace_diff", "Fg3Pct_diff", "Fg2Pct_diff", "TsPct_diff",
            "OffRtg_diff", "DefRtg_diff", "NetRtg_diff", "EfgDiff_diff", "TsDiff_diff",
            "Rebounds_diff", "Steals_diff", "Blocks_diff",

            # Home-court context
            "IsHome", "HomeAdvantage",
        ]

        load_dotenv()
        try:
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_KEY"),
            )
        except Exception as e:
            raise CustomException(f"Failed to initialize Supabase client: {e}", sys)

    # -------------------------------------------------------------------------
    def _initialize_model(self):
        """Initializes the chosen model type (currently only LogisticRegression)."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
                n_jobs=-1,
            )
        else:
            raise CustomException(f"Unsupported model type: {self.model_type}", sys)

    # -------------------------------------------------------------------------
    def train_model(self, data_key="clean/training_data_clean.csv", save=True):
        """
        Trains a general-purpose classifier on the cleaned dataset
        and optionally uploads it (model + scaler) to Supabase.
        """
        try:
            # -----------------------------
            # 📥 Load training data
            # -----------------------------
            print(f"⬇️ Downloading training data from Supabase: {data_key}")
            res = self.supabase.storage.from_(self.bucket_name).download(data_key)
            df = pd.read_csv(io.BytesIO(res))

            if "TeamWin" not in df.columns:
                raise CustomException("Training data missing 'TeamWin' target column.", sys)

            available_features = [f for f in self.features if f in df.columns]
            missing = [f for f in self.features if f not in df.columns]
            if missing:
                print(f"⚠️ Warning: {len(missing)} feature(s) missing, skipping: {missing}")

            X = df[available_features]
            y = df["TeamWin"]
            print(f"📊 Loaded dataset with {X.shape[0]} samples × {X.shape[1]} features")

            # -----------------------------
            # ✂️ Train/Test Split
            # -----------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
            )

            # -----------------------------
            # ⚖️ Standardization
            # -----------------------------
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # -----------------------------
            # 🧠 Initialize & Train Model
            # -----------------------------
            model = self._initialize_model()
            print(f"🚀 Training {self.model_type.replace('_', ' ').title()}...")
            model.fit(X_train_scaled, y_train)

            # -----------------------------
            # 📈 Evaluate Model
            # -----------------------------
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)

            print("\n✅ Model Performance Summary:")
            print(f"   Accuracy: {acc:.4f}")
            print(f"   ROC-AUC:  {auc:.4f}")
            print(classification_report(y_test, y_pred))

            # -----------------------------
            # ☁️ Save to Supabase
            # -----------------------------
            if save:
                model_obj = {"model": model, "scaler": scaler}
                tmp_path = "prediction_model.skops"
                sio.dump(model_obj, tmp_path)

                with open(tmp_path, "rb") as f:
                    file_bytes = f.read()

                model_key = "models/prediction_model.skops"
                print(f"⬆️ Uploading trained model to Supabase → {model_key}")

                res = self.supabase.storage.from_(self.bucket_name).upload(
                    path=model_key,
                    file=file_bytes,
                    file_options={"content_type": "application/octet-stream", "upsert": "true"},
                )

                if hasattr(res, "error") and res.error:
                    raise CustomException(f"Supabase upload failed: {res.error}", sys)

                print("💾 Model and scaler uploaded successfully.")

            return model, scaler

        except Exception as e:
            raise CustomException(f"train_model failed: {e}", sys)
