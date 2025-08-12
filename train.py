import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import os

def load_data(file_path):
    """Loads the credit card default dataset."""
    # Assuming the data is in a local CSV file.
    # For a real project, this might load from S3.
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocesses the data by handling categorical features."""
    # The dataset has many features, we'll select a few for a simple model.
    # The 'default.payment.next.month' column is our target variable.
    df.columns = df.columns.str.replace('.', '_')
    df.rename(columns={'default_payment_next_month': 'target'}, inplace=True)
    
    # We will use simple feature selection for this example
    features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
    
    X = df[features]
    y = df['target']
    
    # For simplicity, we won't do complex scaling or one-hot encoding here,
    # but a real-world model would require it.
    
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
        
        # Log the model
        mlflow.sklearn.log_model(model, "random-forest-model")
        
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