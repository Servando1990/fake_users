# Fake users detection. Take home project

## Features

- Command-line interfaces for training and prediction
- REST API for real-time predictions
- Logistic regression model with balanced class weights
- Feature engineering focused on:
  - Click patterns and ratios
  - Category interaction diversity
  - Event type distribution
- Cross-validation and model evaluation
- Logging system

## How to run

1. Clone this repository
2. Create and activate conda environment:
   ```
   conda env create -f environment.yml
   conda activate fake_user_detection
   ```

## Project Structure

```
.
├── README.md
├── environment.yml        # Conda environment specification
├── data/                # Data directory
│   └── test_data.csv    # Generated test dataset
├── models/              # Model artifacts
│   ├── fake_user_detector.joblib  # Trained model
│   └── training_metrics.json      # Training metrics
└── src/
    ├── main.py          # CLI and API entry points
    ├── train.py         # Training pipeline
    ├── preprocessing/   # Data preprocessing modules
    │   └── processor.py
    ├── model/          # Model-related modules
    │   └── detector.py
    └── utils/          # Utility functions
        └── logger.py   # Logging configuration
```

## Usage

### 1. Training a Mod
Basic usage with default parameters:
```
python src/train.py
```

Customize training parameters:
```
python src/train.py --input data/fake_users.csv --model-output models/my_model.joblib --metrics-output models/my_metrics.json --test-size 0.25 --cv-folds 5 --random-state 42
```

Available training arguments:
- ```--input```: Path to input CSV data (default: fake_users.csv)
- ```--model-output```: Path to save trained model (default: models/fake_user_detector.joblib)
- ```--metrics-output```: Path to save metrics (default: models/training_metrics.json)
- ```--test-size```: Proportion of data for testing (default: 0.2)
- ```--cv-folds```: Number of cross-validation folds (default: 5)
- ```--random-state```: Random seed for reproducibility (default: 42)

### 2. Making Predictions

#### Using CLI:

```
python src/main.py predict --model models/fake_user_detector.joblib --data test_data.csv --output predictions.csv
```

#### Using API:

1. Start the server:

```
python src/main.py serve --port 8000
```

2. Make predictions via HTTP POST:

```
curl -X POST "http://localhost:8000/predict" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@test_data.csv"
```

## Input Data Format

The input CSV files should contain the following columns:

- UserId: unique identifier for each user
- Event: the action performed by the user (e.g., click_ad, send_email)
- Category: the category the user interacted with
- Fake: (only for training data) 1 for fake users, 0 for real users

## Output Format

### Training Output

- Trained model saved in joblib format
- Metrics JSON file containing:
  - Cross-validation F1 scores
  - Training metrics
  - Test set metrics
- Test dataset for evaluation

### Prediction Output

CSV file with three columns:
- UserId: the user identifier
- Fake: predicted label (1 for fake users, 0 for real users)

## Logs

Logs are stored in the `logs` directory:

- `all.log`: Contains all log levels
- `error.log`: Contains only error messages


## Future Improvements

1. **Abstract Class Implementation**


2. **Docker Containerization**

3. **Metaflow Integration**

 - We can use @resources decoratorfor scalable compute
 - also we can simplify dependency management across environments

4. **CI/CD Pipeline**
