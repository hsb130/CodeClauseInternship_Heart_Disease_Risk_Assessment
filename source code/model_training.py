from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib
from data_transformation import transform_data

def train_and_save_model(x_train, y_train, x_test, y_test, model_path='Heart Disease Predictor Model.joblib'):
    """
    Create, train, and save a pipeline using scikit-learn.

    Args:
    - x_train: Training features.
    - y_train: Training labels.
    - x_test: Testing features.
    - y_test: Testing labels.
    - model_path: File path to save the trained model.

    Returns:
    - pipeline: Trained scikit-learn Pipeline.
    """
    # Create the preprocessor
    _, _, preprocessor = transform_data(x_train, y_train, x_test)
    
    # Create the SVM model
    model = SVC()
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classification', model)
    ])
    
    # Train the pipeline
    pipeline.fit(x_train, y_train)
    
    # Save the pipeline
    joblib.dump(pipeline, model_path)
    
    return pipeline
