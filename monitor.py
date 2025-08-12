import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from sklearn.model_selection import train_test_split
import os

# Assuming you have your training data and new data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Load the training data (your reference data)
training_data = load_data('default_of_credit_card_clients.csv')
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
            'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
training_data = training_data.rename(columns={'default.payment.next.month': 'target'})
training_data = training_data[features] # Use the same features as your model

# Simulate new production data (your current data)
# For this example, we'll just split the original data
_, production_data = train_test_split(training_data, test_size=0.1, random_state=42)

# Create a data quality and data drift report
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])
data_drift_report.run(reference_data=training_data, current_data=production_data, column_mapping=None)

# Save the report as an HTML file
data_drift_report.save_html("data_drift_report.html")
print("Data drift report saved to data_drift_report.html")