from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from pathlib import Path
from loguru import logger
import argparse
import json

from preprocessing.processor import DataProcessor
from model.detector import FakeUserDetector

def prepare_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dict, str, DataProcessor]:
    """
    Prepare and split data for training and testing.
    
    Args:
        data_path (str): Path to the raw data CSV
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[Dict, str, DataProcessor]: Dictionary with data splits, test data path, and fitted processor
    """
    # Load and process data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    processor = DataProcessor()
    
    # Process full dataset
    features, user_ids = processor.process_data(df, is_training=True)  # This fits the scaler
    labels = df.groupby('UserId')['Fake'].first().values
    
    # Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        features, labels, user_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Create test dataset for inference testing (without Fake column)
    test_data = []
    for user_id in ids_test:
        user_events = df[df['UserId'] == user_id][['UserId', 'Event', 'Category']]
        test_data.append(user_events)
    
    test_df = pd.concat(test_data, axis=0)
    
    # Save test data
    test_data_path = 'test_data.csv'
    Path('data').mkdir(exist_ok=True)
    test_df.to_csv(test_data_path, index=False)
    
    # Save ground truth for evaluation
    ground_truth = pd.DataFrame({
        'UserId': ids_test,
        'Fake': y_test
    })
    ground_truth.to_csv('data/test_data_ground_truth.csv', index=False)
    
    logger.info(f"Created test set with {len(ids_test)} users and {len(test_df)} events")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'user_ids_test': ids_test
    }, test_data_path, processor

def train_and_evaluate(data_splits: Dict, model_output_path: str, cv_folds: int = 5) -> Dict:
    """
    Train model with cross-validation and final evaluation.
    
    Args:
        data_splits (Dict): Dictionary containing data splits
        model_output_path (str): Where to save the trained model
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        Dict: Dictionary containing all evaluation metrics
    """
    X_train = data_splits['X_train']
    X_test = data_splits['X_test']
    y_train = data_splits['y_train']
    y_test = data_splits['y_test']
    
    # Initialize model
    model = FakeUserDetector()
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model.model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring='f1'
    )
    
    logger.info(f"Cross-validation F1 scores: {cv_scores}")
    logger.info(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    train_metrics = model.train(X_train, y_train)
    
    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_metrics = classification_report(y_test, test_predictions, output_dict=True)
    
    # Save model and scaler
    model.save_model(model_output_path)
    
    return {
        'cross_validation': {
            'f1_scores': cv_scores.tolist(),
            'mean_f1': float(cv_scores.mean()),
            'std_f1': float(cv_scores.std())
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train fake user detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/fake_users.csv',
        help='Path to input CSV data'
    )
    
    parser.add_argument(
        '--model-output',
        type=str,
        default='models/fake_user_detector.joblib',
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--metrics-output',
        type=str,
        default='models/training_metrics.json',
        help='Path to save training metrics'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger.info(f"Starting training with parameters:")
    logger.info(f"Input data: {args.input}")
    logger.info(f"Model output: {args.model_output}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"CV folds: {args.cv_folds}")
    
    # Prepare data and get test set path
    data_splits, test_data_path, processor = prepare_data(
        args.input,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Train and evaluate
    metrics = train_and_evaluate(
        data_splits,
        args.model_output,
        cv_folds=args.cv_folds
    )
    
    # Save the fitted scaler
    scaler_path = str(Path(args.model_output).parent / 'scaler.joblib')
    processor.save_scaler(scaler_path)
    logger.info(f"Saved fitted scaler to {scaler_path}")
    
    # Save metrics
    Path(args.metrics_output).parent.mkdir(exist_ok=True)
    with open(args.metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {args.model_output}")
    logger.info(f"Metrics saved to {args.metrics_output}")
    logger.info(f"Test data saved to {test_data_path}")
    logger.info(f"Ground truth saved to data/test_data_ground_truth.csv")
    logger.info("\nYou can now test the model using either:")
    logger.info(f"1. CLI: python src/main.py predict --model {args.model_output} --data {test_data_path} --output predictions.csv")
    logger.info("2. API: python src/main.py serve --port 8000")