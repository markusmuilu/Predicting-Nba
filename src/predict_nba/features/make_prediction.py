"""
High-level orchestration module for NBA predictions.
Coordinates data collection, cleaning, training, and matchup prediction.
"""
import sys 

from predict_nba.features.data_collector import DataCollector
from predict_nba.features.data_cleaner import DataCleaner
from predict_nba.features.model_predictor import ModelPredictor
from predict_nba.features.model_trainer import ModelTrainer
from predict_nba.utils.logger import logger
from predict_nba.utils.exception import CustomException


class MakePrediction:
    """
    Central class orchestrating data collection, cleaning, and model prediction.
    Used by both CLI tools and API endpoints.
    """

    def predict(self, team1: str, team2: str):
        """
        Predicts the winner between two teams.
        Returns a dict: {"winner": str, "confidence": float}
        """
        try:
            logger.info(f"Starting prediction for {team1} vs {team2}")

            # Collect latest data for each team
            DataCollector().get_current_season(team1)
            DataCollector().get_current_season(team2)

            # Clean and prepare prediction dataset
            DataCleaner().clean_prediction_data(team1, team2)

            # Run actual model prediction
            result = ModelPredictor().predict_matchup(team1, team2)

            if result is None:
                logger.warning("Prediction returned None.")
                return None

            logger.info(f"Prediction complete: {result}")
            return result

        except Exception as e:
            CustomException(f"predict() failed: {e}", sys)
            return None

    def train(self, seasons: list[str]):
        """
        Trains a new model using the provided seasons list.
        Example: ["2022-23", "2023-24"]
        """
        try:
            logger.info(f"Starting model training for seasons: {seasons}")

            collector = DataCollector()
            cleaner = DataCleaner()
            trainer = ModelTrainer()

            # Step 1: Collect raw training data
            collector.collect_training_data(seasons)

            # Step 2: Clean training dataset
            cleaner.clean_training_data(seasons)

            # Step 3: Train and upload the ML model
            trainer.train_model()

            logger.info("Model training workflow completed successfully.")

        except Exception as e:
            CustomException(f"train() failed: {e}", sys)
            return None

