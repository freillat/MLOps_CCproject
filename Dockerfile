# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install git and other build dependencies
RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
COPY . .

# Set an environment variable for MLflow tracking (if needed)
# ENV MLFLOW_TRACKING_URI=http://<your-mlflow-server-uri>:5000

# The command to run the Prefect flow when the container starts
CMD ["python", "flow.py"]