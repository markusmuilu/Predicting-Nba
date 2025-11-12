# src/features/predictor.py
from src.features.data_collector import DataCollector
from src.features.data_cleaner import DataCleaner
from src.features.model_predictor import ModelPredictor
from src.features.model_trainer import ModelTrainer

class MakePrediction:
    """
    Central class orchestrating data collection, cleaning, and model prediction.
    Used by both CLI and API endpoints.
    """

    def predict(self, team1: str, team2: str):
        """
        Predicts the winner between two teams.
        Returns: {"winner": str, "confidence": float}
        """

        DataCollector().get_current_season(team1)
        DataCollector().get_current_season(team2)
        DataCleaner().clean_prediction_data(team1, team2)

        return ModelPredictor().predict_matchup(team1, team2)

    def train(self, seasons: list[str]):
        """
        Trains a new model using the provided seasons.
        """
        collector = DataCollector()
        cleaner = DataCleaner()
        trainer = ModelTrainer()

        collector.collect_training_data(seasons)
        cleaner.clean_training_data(seasons)
        trainer.train_model(seasons)
