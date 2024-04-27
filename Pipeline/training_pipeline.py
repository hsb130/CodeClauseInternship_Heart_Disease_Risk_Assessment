# Importing necessary modules
import data_collection
import train_test_split
import model_training

if __name__ == '__main__':
    # Specify the file path for the data
    file_path = "E:\Hasib's Github\Data Science Intern@CodeClause\Task-4 Heart Disease Risk Assessment\heart.csv"
    
    # Load data
    df = data_collection.load_data(file_path)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split.split_data(df)
    
    # Train the model and save the pipeline
    pipeline = model_training.train_and_save_model(x_train, y_train, x_test, y_test)
