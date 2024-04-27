import joblib
import pandas as pd

def predict(file_path, model_path):
    """
    Predict using a trained pipeline on unseen data.

    Args:
    - file_path: Path to the CSV file containing unseen data.
    - model_path: Path to the joblib file containing the trained pipeline.

    Returns:
    - predictions: Predictions for the unseen data.
    """
    # Load the unseen data
    unseen_data = pd.read_csv(file_path)

    # Load the trained pipeline
    pipeline = joblib.load(model_path)

    # Make predictions
    predictions = pipeline.predict(unseen_data)

    return predictions

if __name__ == '__main__':
    # Specify the file path for the unseen data and the model
    unseen_data_path = r"E:\Hasib's Github\Data Science Intern@CodeClause\Task-4 Heart Disease Risk Assessment\unseen.csv"
    model_path = 'Heart Disease Predictor Model.joblib'
    
    # Make predictions
    predictions = predict(unseen_data_path, model_path)
    print(predictions)
