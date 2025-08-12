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




# How to guide

This document provides a step-by-step guide to setting up and running the end-to-end MLOps project from scratch. It assumes you have access to the codebase and a configured AWS account.

-----

## 1\. Prerequisites 📋

Before you begin, ensure your local environment has the following:

  * **Python:** The project requires Python 3.8 or higher.

  * **Dependencies:** All necessary Python libraries can be installed with the provided `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

  * **AWS CLI:** Configure your local machine with access to your AWS account.

    ```bash
    aws configure
    ```

  * **Terraform:** This project uses Terraform for Infrastructure as Code (IaC). Ensure it is installed on your system.

-----

## 2\. Project Setup 🚀

The first step is to provision the necessary cloud infrastructure and configure your environment.

1.  **Provision AWS Resources with Terraform:**

      * Navigate to the directory containing the `main.tf` file.
      * Initialize Terraform and apply the configuration to create an S3 bucket for data and model storage.

    <!-- end list -->

    ```bash
    # Replace "your-unique-mlops-project-bucket-name" with a globally unique name
    terraform init
    terraform apply
    ```

2.  **Upload Data to S3:**

      * Upload the original `default_of_credit_card_clients.csv` dataset to the root of your new S3 bucket.
      * Upload a new CSV file (e.g., `new_customer_data.csv`) that simulates unseen data for the batch prediction pipeline.

3.  **Start the MLflow UI:**

      * Open a new terminal and start the MLflow tracking server. This will enable experiment logging.

    <!-- end list -->

    ```bash
    mlflow ui
    ```

      * The UI is accessible at `http://localhost:5000`.

4.  **Configure Prefect S3 Block:**

      * In the Prefect UI, create an **S3Bucket** block with the name you used in the `batch_flow.py` script. This block securely stores your AWS credentials and bucket name for the Prefect flows.

-----

## 3\. Running the Pipeline ⚙️

Follow these steps to execute the training, deployment, and monitoring phases of the project.

### **Phase 1: Training and Experiment Tracking**

  * Execute the Prefect flow that orchestrates the training script. This script will preprocess data, train a **Random Forest Classifier**, and log the experiment to MLflow.

    ```bash
    python flow.py
    ```

  * After the flow completes, navigate to the MLflow UI to view the experiment run. **Note down the Run ID**, as it is needed for the deployment step.

### **Phase 2: Batch Prediction and Monitoring**

  * Open the `batch_flow.py` script and update the `mlflow_model_uri` with the **Run ID** you noted in the previous step.

  * Run the batch prediction flow, which will load the trained model, make predictions on the new data from S3, save the results back to S3, and generate an Evidently report.

    ```bash
    python batch_flow.py
    ```

  * Once the flow finishes, you will find a new `data_drift_report.html` file in your local project directory. Open this file to review the data drift analysis.

-----

## 4\. Conclusion 🎉

By following these steps, you will have successfully run an end-to-end MLOps pipeline that demonstrates:

  * **Workflow Orchestration:** Automated training and prediction using Prefect.
  * **Experiment Tracking:** Versioning models and metrics with MLflow.
  * **Batch Deployment:** Serving predictions on new data from S3.
  * **Monitoring:** Detecting data drift with Evidently.
  * **Infrastructure as Code:** Managing cloud resources with Terraform.