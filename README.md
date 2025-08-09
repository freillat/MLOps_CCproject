# End-to-End MLOps Project: Credit Card Default Prediction
This repository contains an end-to-end Machine Learning Operations (MLOps) project focused on predicting credit card default. The goal of this project is to demonstrate the full lifecycle of an ML model, from data preparation and training to deployment and monitoring, using a set of simple yet powerful MLOps tools.

<br>

1. Problem Statement
Credit card default prediction is a binary classification problem that aims to identify customers who are likely to fail on their credit card payments. This project builds a machine learning model to tackle this challenge, which is critical for financial institutions in managing risk and making informed credit decisions. The project focuses on a complete, reproducible, and automated pipeline that can be easily adapted to new data and model versions.

<br>

2. Dataset
The project uses the Taiwanese Credit Card Default Dataset, a public dataset widely used in machine learning and data science. It contains various features related to customer demographics, credit history, and payment behavior. The target variable is default payment next month, which is a binary indicator (1 for default, 0 for no default). This dataset provides a solid foundation for training a robust classification model and simulating real-world scenarios.

<br>

3. How the Elements Work Together
The project is structured around a modular and automated pipeline that covers the key stages of an MLOps lifecycle. The following tools are integrated to achieve this:

⚙️ Technologies
Cloud: AWS for infrastructure and data storage (S3).

ML Framework: Scikit-learn for building the classification model.

Experiment Tracking: MLflow to log and manage model experiments, metrics, and artifacts.

Workflow Orchestration: Prefect to automate the training and batch prediction pipelines.

Monitoring: Evidently to generate reports on data drift and model performance.

CI/CD: GitHub Actions to automate code testing.

IaC: Terraform to manage the cloud infrastructure.

🧩 Workflow
Data Ingestion & Training: The train.py script preprocesses the credit card data and trains a Random Forest Classifier. All experiments, including model parameters and performance metrics (e.g., accuracy, F1-score), are logged to MLflow. This process is orchestrated by a Prefect Flow.

Deployment (Batch): A separate script (not shown here) would load the best model from MLflow and perform predictions on new data. This batch prediction job is also managed as a Prefect Flow, running on a schedule or triggered by an event.

Monitoring: After a batch prediction run, an Evidently report is generated to compare the new production data with the original training data. This helps detect data drift and ensures the model's inputs remain consistent.

CI/CD: A GitHub Actions workflow runs pytest every time code is pushed to the main branch. This ensures that the data preprocessing and training logic are always working as expected before any changes are merged.

Infrastructure: All cloud resources, such as the S3 bucket for data storage, are defined using Terraform. This allows for the entire infrastructure to be provisioned and updated in a reproducible way.

This architecture creates a reliable and automated system for building and maintaining a production-ready machine learning model.