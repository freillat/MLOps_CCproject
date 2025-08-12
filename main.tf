terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = "us-east-1"  # Or your preferred region
}

# Create an S3 bucket for ML project assets
resource "aws_s3_bucket" "mlops_project_bucket" {
  bucket = "your-unique-mlops-project-bucket-name" # Must be globally unique!
  
  tags = {
    Project = "MLOps-Credit-Default"
    Environment = "Dev"
  }
}

# Enable versioning on the S3 bucket (best practice for data/models)
resource "aws_s3_bucket_versioning" "mlops_project_bucket_versioning" {
  bucket = aws_s3_bucket.mlops_project_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}