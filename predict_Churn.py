import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads Churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath)  # Use the argument filepath instead of hardcoding
    return df

def make_predictions(df):
    """
    Uses the PyCaret best model to make predictions on data in the df DataFrame.
    """
    model = load_model('KNeighborsClassifier')  # Ensure this model exists
    predictions = predict_model(model, data=df)

    # Check the column names
    print("Columns in predictions DataFrame:", predictions.columns)
    
    # Rename 'prediction_label' to 'Churn_prediction' if it exists
    if 'prediction_label' in predictions.columns:
        predictions.rename(columns={'prediction_label': 'Churn_prediction'}, inplace=True)
        
        # Replace values in the new column
        predictions['Churn_prediction'].replace({1: 'Yes', 0: 'No'}, inplace=True)
        
        return predictions['Churn_prediction']
    else:
        raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame")

if __name__ == "__main__":
    # Use raw string (r"") or double backslashes to fix file path issue
    file_path = r"C:\Datascience\prepared_churn_data (1).csv"  
    df = load_data(file_path)  
    predictions = make_predictions(df)

    print('Predictions:')
    print(predictions)
