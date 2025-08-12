import pandas as pd
import numpy as np
import prefect
from prefect import flow, task
from prefect_aws.s3 import S3Bucket
import mlflow
from mlflow.tracking import MlflowClient
import os
import pickle
from datetime import datetime

# Import Evidently for monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset # Use RegressionPreset for regression, we'll need a different one for classification

# --- Task 1: Loading Data from S3 ---
@task
def load_data(bucket_name: str, file_path: str) -> pd.DataFrame:
    """Loads a DataFrame from an S3 bucket."""
    s3_bucket = S3Bucket.load("your-s3-bucket-block-name") # Replace with your Prefect Block name
    
    s3_path = f"s3://{bucket_name}/{file_path}"
    print(f"Reading data from {s3_path}...")
    
    # Using pandas' native S3 support with fsspec and boto3
    try:
        df = pd.read_csv(s3_path)
        print(f"Data loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        return pd.DataFrame()

# --- Task 2: Preprocessing and Making Predictions ---
@task
def predict(df: pd.DataFrame, model_uri: str) -> pd.DataFrame:
    """Loads the model and makes predictions on new data."""
    if df.empty:
        print("Input DataFrame is empty. Skipping prediction.")
        return pd.DataFrame()
    
    # Load the model from MLflow's Model Registry
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from {model_uri}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return pd.DataFrame()

    # Preprocessing the new data
    df.columns = df.columns.str.replace('.', '_')
    features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
    X_pred = df[features]
    
    # Make predictions
    predictions = model.predict(X_pred)
    df['prediction'] = predictions
    print(f"Predictions made for {len(df)} records.")
    
    return df

# --- Task 3: Saving Predictions to S3 ---
@task
def save_predictions(df: pd.DataFrame, bucket_name: str, file_path: str):
    """Saves the DataFrame with predictions back to S3."""
    if df.empty:
        print("Prediction DataFrame is empty. Skipping save.")
        return

    s3_path = f"s3://{bucket_name}/{file_path}"
    print(f"Saving predictions to {s3_path}...")
    
    try:
        # Use pandas to save to Parquet on S3
        df.to_parquet(s3_path, index=False)
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions to S3: {e}")

# --- Task 4: Monitoring with Evidently ---
@task
def monitor_data_drift(reference_data_path: str, current_data: pd.DataFrame):
    """Generates an Evidently data drift report."""
    if current_data.empty:
        print("Current data is empty. Skipping monitoring.")
        return

    try:
        # Load the reference data (e.g., training data) from S3
        reference_data = pd.read_csv(reference_data_path)
        print(f"Reference data loaded with shape: {reference_data.shape}")

        # Ensure reference data has the same feature set
        reference_data.columns = reference_data.columns.str.replace('.', '_')
        features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                    'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
        reference_data = reference_data[features]
        current_data = current_data[features] # Monitor only the features
        
        # Create an Evidently report
        data_drift_report = Report(metrics=[
            DataDriftPreset()
        ])
        
        data_drift_report.run(
            reference_data=reference_data, 
            current_data=current_data, 
            column_mapping=None
        )
        
        # Save the report with a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        report_filename = f"data_drift_report_{timestamp}.html"
        data_drift_report.save_html(report_filename)
        print(f"Evidently data drift report saved as {report_filename}")

    except Exception as e:
        print(f"Error during Evidently monitoring: {e}")

# --- The Main Prefect Flow ---
@flow(name="Batch Prediction and Monitoring")
def batch_prediction_flow(
    input_file: str,
    output_file: str,
    reference_data_path: str,
    mlflow_model_uri: str,
    bucket_name: str = "your-unique-mlops-project-bucket-name"
):
    """Orchestrates the batch prediction and monitoring pipeline."""
    
    # 1. Load new data for prediction
    new_data = load_data(bucket_name, input_file)

    # 2. Make predictions
    predicted_df = predict(new_data, mlflow_model_uri)

    # 3. Save predictions
    save_predictions(predicted_df, bucket_name, output_file)

    # 4. Monitor the new data for drift
    monitor_data_drift(reference_data_path, new_data)

if __name__ == "__main__":
    # Example usage with placeholder values
    # These would be passed as parameters in a real deployment
    # For local testing, you'd need to create dummy files in S3 first
    input_file_path = "new_customer_data_february.csv"
    output_file_path = "predictions/february_predictions.parquet"
    reference_data_s3_path = "default_of_credit_card_clients.csv"
    # The URI of the model you registered in MLflow
    model_uri = "runs:/<YOUR_MLFLOW_RUN_ID>/random-forest-model" 
    
    batch_prediction_flow(
        input_file=input_file_path, 
        output_file=output_file_path,
        reference_data_path=reference_data_s3_path,
        mlflow_model_uri=model_uri
    )