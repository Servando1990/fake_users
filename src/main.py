import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Union
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
from preprocessing.processor import DataProcessor
from model.detector import FakeUserDetector
from utils.logger import setup_logging
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="Fake User Detection API",
    description="API for detecting fake users based on their behavior patterns",
    version="1.0.0"
)

class PredictionResponse(BaseModel):
    user_id: str
    fake_probability: float
    is_fake: bool

def train_model(data_path: str, model_output_path: str) -> Dict:
    """
    Train the fake user detection model and save it.
    
    Args:
        data_path (str): Path to the training data CSV
        model_output_path (str): Where to save the trained model
        
    Returns:
        Dict: Training metrics
    """
    logger.info(f"Starting model training with data from {data_path}")
    
    # Load and process data
    df = pd.read_csv(data_path)
    processor = DataProcessor()
    features, user_ids = processor.process_data(df, is_training=True)
    labels = df.groupby('UserId')['Fake'].first().values
    
    # Train and save model
    model = FakeUserDetector()
    metrics = model.train(features, labels)
    model.save_model(model_output_path)
    
    return metrics

def predict(model_path: str, data_path: str, output_path: str) -> None:
    """
    Make predictions on new data using a trained model.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the test data CSV
        output_path (str): Where to save predictions
    """
    logger.info(f"Starting prediction using model from {model_path}")
    
    # Load and process data
    df = pd.read_csv(data_path)
    processor = DataProcessor()
    
    # Load the scaler
    scaler_path = str(Path(model_path).parent / 'scaler.joblib')
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please ensure you have trained the model first.")
    processor.load_scaler(scaler_path)
    
    # Process data
    features, user_ids = processor.process_data(df, is_training=False)
    
    # Load model and make predictions
    model = FakeUserDetector(model_path)
    predictions = model.predict(features)

    
    # Save predictions in required format
    results = pd.DataFrame({
        'UserId': user_ids,
        'Fake': predictions.astype(int)  # Ensure predictions are 0/1
    })
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # If ground truth exists, evaluate predictions
    ground_truth_path = 'data/test_data_ground_truth.csv'
    if Path(ground_truth_path).exists():
        ground_truth = pd.read_csv(ground_truth_path)
        merged = results.merge(ground_truth, on='UserId')
        accuracy = (merged['Fake_x'] == merged['Fake_y']).mean()
        logger.info(f"Prediction accuracy on test set: {accuracy:.3f}")

@app.post("/predict", response_model=List[Dict[str, Union[str, int]]])
async def predict_api(file: UploadFile = File(...), model_path: str = "models/fake_user_detector.joblib"):
    """API endpoint for making predictions in the same format as predictions.csv"""
    # Save uploaded file temporarily
    temp_path = "temp_data.csv"
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Initialize processor and load scaler
    processor = DataProcessor()
    scaler_path = str(Path(model_path).parent / 'scaler.joblib')
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please ensure you have trained the model first.")
    processor.load_scaler(scaler_path)
    
    # Process data and make predictions
    df = pd.read_csv(temp_path)
    features, user_ids = processor.process_data(df, is_training=False)
    
    model = FakeUserDetector(model_path)
    predictions = model.predict(features)
    
    # Clean up
    Path(temp_path).unlink()
    
    # Format response to match predictions.csv format
    return [
        {
            "UserId": user_id,
            "Fake": int(pred)  # Convert to int to match CSV format
        }
        for user_id, pred in zip(user_ids, predictions)
    ]

def main():
    # Setup logging
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Fake User Detection CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--data', required=True, help='Path to test data CSV')
    predict_parser.add_argument('--output', required=True, help='Where to save predictions CSV')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the prediction API server')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    serve_parser.add_argument('--model', type=str, default='models/fake_user_detector.joblib', 
                            help='Path to model file to use for predictions')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        predict(args.model, args.data, args.output)
    elif args.command == 'serve':
        logger.info(f"Starting API server on port {args.port}")
        logger.info(f"Using model: {args.model}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 