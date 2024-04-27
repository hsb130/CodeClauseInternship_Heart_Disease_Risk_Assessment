from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def transform_data(x_train, y_train, x_test):
    """
    Preprocess data using a ColumnTransformer.

    Args:
    - x_train: Training features.
    - y_train: Training labels.
    - x_test: Testing features.

    Returns:
    - x_train_preprocessed: Preprocessed training features.
    - x_test_preprocessed: Preprocessed testing features.
    - preprocessor: ColumnTransformer used for preprocessing.
    """
    # Identify categorical and numerical columns directly from x_train
    categorical_columns = x_train.select_dtypes(include=['object']).columns
    numerical_columns = x_train.select_dtypes(include=['number']).columns

    # Create a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='first'), categorical_columns),
            ('lda', LDA(n_components=1), numerical_columns)
        ])

    # Fit and transform the training set
    x_train_preprocessed = preprocessor.fit_transform(x_train, y_train)

    # Transform the test set
    x_test_preprocessed = preprocessor.transform(x_test)

    return x_train_preprocessed, x_test_preprocessed, preprocessor
