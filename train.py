import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
from mlflow.models import infer_signature
import os

def load_data(file_path):
    """Loads the credit card default dataset."""
    # Assuming the data is in a local CSV file.
    # For a real project, this might load from S3.
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocesses the data by handling categorical features."""
    
    # 1. Normalize column names
    df.columns = df.columns.str.replace('.', '_', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    
    # 2. Check for and rename the target column
    original_target_col = 'default_payment_next_month'
    if original_target_col not in df.columns:
        raise KeyError(f"Column '{original_target_col}' not found.")
    df.rename(columns={original_target_col: 'target'}, inplace=True)
    
    # 3. Handle missing values
    # Drop rows where the target variable 'target' is missing
    df.dropna(subset=['target'], inplace=True)
    
    features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
    
    X = df[features]
    y = df['target']
    
    # Optional: You might also want to handle missing values in the features (X)
    # For this simple model, you could drop rows with any missing feature values
    # df.dropna(subset=features, inplace=True)
    
    return X, y

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Trains a RandomForestClassifier and logs to MLflow."""
    mlflow.set_experiment("Credit Card Default Prediction")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        recall = recall_score(y_train, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Get a sample of the training data
        sample_input = X_train.sample(n=5, random_state=42)
                
        integer_cols = sample_input.select_dtypes(include=['int64']).columns.tolist()
        
        if integer_cols:
            sample_with_nan = sample_input.copy()
            # Add a NaN to the first integer column
            sample_with_nan.loc[sample_with_nan.index[0], integer_cols[0]] = pd.NA
        else:
            # If no integer columns are found, use the original sample
            sample_with_nan = sample_input

        # Infer the model signature using the sample with a NaN
        signature = infer_signature(
            model_input=sample_with_nan, 
            model_output=model.predict(sample_input)
        )
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random-forest-model-artifact",
            signature=signature,
            input_example=sample_with_nan
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")
        
    return model

if __name__ == '__main__':
    # You'll need to download the dataset and place it in your project folder
    # e.g., default_of_credit_card_clients.csv
    data_path = 'default_of_credit_card_clients.csv'
    
    df = load_data(data_path)
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = train_model(X_train, y_train)