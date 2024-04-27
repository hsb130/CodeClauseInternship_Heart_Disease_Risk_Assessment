from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.1, random_state=42):
    """
    Split data into training and testing sets.

    Args:
    - df: DataFrame containing the dataset.
    - test_size: Fraction of data to use as the test set.
    - random_state: Seed for random number generation.

    Returns:
    - x_train: Training features.
    - x_test: Testing features.
    - y_train: Training labels.
    - y_test: Testing labels.
    """
    x = df.drop(columns='HeartDisease')
    y = df['HeartDisease']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    return x_train, x_test, y_train, y_test
