from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib
from pathlib import Path

class DataProcessor:
    """Handles preprocessing of user event data for fake user detection."""
    
    def __init__(self) -> None:
        """Initialize the DataProcessor with necessary preprocessing components."""
        self.scaler = StandardScaler()
        logger.info("Initialized DataProcessor")
        
    def save_scaler(self, path: str) -> None:
        """Save the fitted scaler to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Saved scaler to {path}")
    
    def load_scaler(self, path: str) -> None:
        """Load a fitted scaler from disk."""
        self.scaler = joblib.load(path)
        logger.info(f"Loaded scaler from {path}")
        
    def process_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Process the input data by creating relevant features for fake user detection.
        
        Args:
            df (pd.DataFrame): Input DataFrame with user events
            is_training (bool): Whether this is training data or inference data
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed features and list of user IDs
        """
        logger.info(f"Processing {'training' if is_training else 'inference'} data")
        
        # Group by UserId and calculate features
        features = df.groupby('UserId').agg({
            'Event': [
                ('total_events', 'count'),
                ('click_ad_ratio', lambda x: (x == 'click_ad').mean()),
                ('send_email_ratio', lambda x: (x == 'send_email').mean())
            ],
            'Category': [
                ('unique_categories', 'nunique'),
                ('category_repeat_ratio', lambda x: len(x) / x.nunique())
            ]
        })
        
        # Flatten column names
        features.columns = features.columns.get_level_values(1)
        features = features.reset_index()
        
        # Calculate event type distribution
        event_counts = df.groupby(['UserId', 'Event']).size().unstack(fill_value=0)
        features = features.merge(event_counts, on='UserId', how='left')
        
        # Store user IDs before dropping the column
        user_ids = features['UserId'].tolist()
        features = features.drop('UserId', axis=1)
        
        # Handle missing values
        features = features.fillna(0)
        
        # Scale features
        if is_training:
            logger.debug("Fitting and transforming features")
            features_scaled = self.scaler.fit_transform(features)
        else:
            logger.debug("Transforming features using pre-fitted scaler")
            features_scaled = self.scaler.transform(features)
            
        return pd.DataFrame(features_scaled, columns=features.columns), user_ids 