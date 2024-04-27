import pandas as pd

def load_data(file_path):
    """
    Load data from the specified file path.

    Args:
    - file_path: Path to the CSV file containing the dataset.

    Returns:
    - df: DataFrame containing the dataset.
    """
    df = pd.read_csv(file_path)
    return df
