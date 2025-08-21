import subprocess
import sys

# --- Install required packages ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3==1.28.57", "botocore==1.31.57", "numpy==1.24.1", "sagemaker" ])

# Ensure sagemaker SDK is installed before importing
try:
    import sagemaker
except ImportError:
    print("sagemaker SDK not found. Installing now...")
    install("sagemaker")
    import sagemaker

import argparse
import sagemaker
import boto3
from sagemaker.model import ModelPackage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept the registered model's ARN instead of the S3 data path
    parser.add_argument("--model-package-arn", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    # Create a SageMaker Model object directly from the Model Package ARN
    model = ModelPackage(
        model_package_arn=args.model_package_arn,
        role=args.role,
        sagemaker_session=sagemaker_session,
    )

    # Deploy the model to an endpoint
    print(f"Deploying registered model from ARN to endpoint: {args.endpoint_name}")
    model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name=args.endpoint_name,
        # Update endpoint if it already exists
        update_endpoint=True
    )
    print("Deployment complete.")
