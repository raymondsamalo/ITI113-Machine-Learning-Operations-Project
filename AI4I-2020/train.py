# # Ensure MLflow is installed
import sys
import subprocess
import pandas as pd
try:
    import mlflow
    import sagemaker_mlflow
except ImportError:
    print("Installing MLflow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                            "boto3==1.37.1", "botocore==1.37.1", "s3transfer",
                            "mlflow==2.22.0", "sagemaker-mlflow==0.1.0"])
import mlflow
import sagemaker_mlflow
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

def train(df, max_depth=5, min_samples_leaf=30, min_samples_split=20):
    # declare our target labels columns
    labels = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(columns=labels)
    y = df[labels]
    model = DecisionTreeClassifier(random_state=42,
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split)
    model = MultiOutputClassifier(model)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, columns=y.columns)
    y_pred_omf = y_pred["Machine failure"]
    y_omf = y["Machine failure"]
    f2 = fbeta_score(y_omf, y_pred_omf, beta=2)
    recall = recall_score(y_omf, y_pred_omf)
    precision = precision_score(y_omf, y_pred_omf, zero_division=0)
    accuracy = accuracy_score(y_omf, y_pred_omf)
    return {
        "model": model,
        "f2": f2,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "input_example": X.head(5)
    }


def run_experiment(df, experiment_name, run_name=None, model_name="model",
                   max_depth=5, min_samples_leaf=30, min_samples_split=20):
    print("Starting experiment ", experiment_name, " with run name ", run_name)
    run_id = None
    # Start an MLflow run
    # Use the experiment name and run name to organize runs
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\tMLflow Run ID : {run_id}")
        print(f"\tRunning experiment: {experiment_name}, Run Name: {run_name}")
        # Train the model
        # Provide the first 5 rows of the training data as an example
        result = train(df,
                       max_depth=max_depth,
                       min_samples_leaf=min_samples_leaf,
                       min_samples_split=min_samples_split)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("f2", result["f2"])
        mlflow.log_metric("accuracy", result["accuracy"])
        mlflow.log_metric("recall", result["recall"])
        mlflow.log_metric("precision", result["precision"])
        model = result["model"]
       # Train the model
        # Provide the first 5 rows of the training data as an example
        input_example = result["input_example"]
        # Get the run ID for later use
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path=model_name, input_example=input_example)
        print("\tFinished: experiment ", experiment_name,
              " with run name ", run_name)
    return model, run_id, result["accuracy"], result["f2"]


if __name__ == "__main__":


    # import mlflow
    # import sagemaker_mlflow
    import mlflow.sklearn
    import os
    import argparse
    import pandas as pd
    import joblib
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_server_arn", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="Default")
    parser.add_argument("--model_output_path", type=str,
                        default="/opt/ml/model")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_leaf", type=int, default=30)
    parser.add_argument("--min_samples_split", type=int, default=20)
    args, _ = parser.parse_known_args()

    # Load training data
    train_path = glob.glob("/opt/ml/input/data/train/*.csv")[0]
    df = pd.read_csv(train_path)
    # Set up MLflow
    mlflow.set_tracking_uri(args.tracking_server_arn)
    the_model, run_id, accuracy_score, f2_score = run_experiment(df=df,
                                                             experiment_name=args.experiment_name,
                                                             run_name="run_name",
                                                             model_name="model",
                                                             max_depth=args.max_depth,
                                                             min_samples_leaf=args.min_samples_leaf,
                                                             min_samples_split=args.min_samples_split
                                                             )
    os.makedirs(args.model_output_path, exist_ok=True)
    joblib.dump(the_model, os.path.join(args.model_output_path, "model.joblib"))
    with open(os.path.join(args.model_output_path, "run_id.txt"), "w") as f:
        f.write(run_id)

    print(
        f"Training complete. F2:{f2_score:.4f} Accuracy: {accuracy_score:.4f}")
    print(f"MLflow Run ID: {run_id}")
