from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from loguru import logger

class FakeUserDetector:
    """Model class for detecting fake users based on their behavior patterns."""
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the fake user detector model.
        
        Args:
            model_path (Optional[str]): Path to a saved model file
        """
        self.model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        if model_path:
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info("Initialized new model instance")
    
    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """
        Train the model on processed features.
        
        Args:
            features (pd.DataFrame): Processed feature matrix
            labels (pd.Series): Binary labels (0: real user, 1: fake user)
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        logger.info(f"Training model on {len(features)} samples")
        self.model.fit(features, labels)
        
        # Calculate training metrics
        predictions = self.model.predict(features)
        metrics = classification_report(labels, predictions, output_dict=True)
        
        logger.info(f"Training accuracy: {metrics['accuracy']:.3f}")
        logger.debug(f"Detailed metrics: {metrics}")
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on processed features.
        
        Args:
            features (pd.DataFrame): Processed feature matrix
            
        Returns:
            np.ndarray: Binary predictions (0: real user, 1: fake user)
        """
        logger.info(f"Making predictions on {len(features)} samples")
        return self.model.predict(features)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for each class.
        
        Args:
            features (pd.DataFrame): Processed feature matrix
            
        Returns:
            np.ndarray: Probability scores for each class
        """
        return self.model.predict_proba(features)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path where to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}") 