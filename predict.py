#!/usr/bin/env python3
"""
predict.py

Load the best logistic regression model from MLflow and make predictions.

Usage examples:

1. Single prediction (inline JSON-like input):
   python predict.py --run-id <run-id> --input '{"age": 42, "job": "technician", ...}'

2. Batch prediction from CSV:
   python predict.py --run-id <run-id> --input-file test_samples.csv --output-file predictions.csv

Requirements:
    pip install mlflow pyspark pandas
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional

import mlflow
import mlflow.pyspark.ml
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


def create_spark_session():
    """Minimal Spark session – enough for vector assembly & model inference"""
    return (
        SparkSession.builder
        .appName("BankMarketing-Prediction")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )


def load_model(run_id: str) -> PipelineModel:
    """
    Load the logged PipelineModel (containing VectorAssembler + StandardScaler + LogisticRegression)
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise ValueError(f"Run {run_id} is not in FINISHED state (status={run.info.status})")

    # MLflow auto-logs the model under this artifact path when using autolog + pyspark.ml
    model_uri = f"runs:/{run_id}/model"

    print(f"Loading model from: {model_uri}")
    model = mlflow.pyspark.ml.load_model(model_uri)
    return model


def prepare_single_input(data_dict: Dict[str, Any], spark: SparkSession):
    """Convert dict → single-row Spark DataFrame with same schema as training"""
    # Convert values to correct types if needed (pandas → spark handles most coercion)
    pdf = pd.DataFrame([data_dict])
    sdf = spark.createDataFrame(pdf)
    return sdf


def prepare_batch_input(csv_path: str, spark: SparkSession):
    """Read CSV → Spark DataFrame"""
    return spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path)


def run_prediction(
    model: PipelineModel,
    data_df,
    output_proba: bool = True,
    output_path: Optional[str] = None
):
    """
    Run model.transform() and extract prediction + probability (if requested)
    """
    prediction_col = "prediction"
    probability_col = "probability"

    result_df = model.transform(data_df)

    # Select useful columns
    output_cols = ["prediction"]
    if output_proba:
        output_cols.append("probability")

    # Keep original input features + prediction columns
    keep_cols = [c for c in data_df.columns] + output_cols
    result = result_df.select(*keep_cols)

    if output_path:
        # Save as CSV (coalesce to 1 file for simplicity)
        result.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
        print(f"Batch predictions saved to: {output_path}")
    else:
        # Show in console (for single / small batch)
        result.show(truncate=False)

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict using MLflow-logged bank marketing model")
    parser.add_argument("--run-id", help="MLflow run ID (from your training notebook)")
    parser.add_argument("--input", type=str, help="JSON string with one example (e.g. '{\"age\":35,...}')")
    parser.add_argument("--input-file", help="Path to CSV file with multiple examples")
    parser.add_argument("--output-file", help="Where to save batch predictions (CSV)")
    parser.add_argument("--no-probability", action="store_true", help="Skip probability column")

    args, unknown = parser.parse_known_args()

    # Check if running in notebook without arguments - provide helpful message
    if not args.run_id:
        print("=" * 70)
        print("NOTE: Running in notebook mode without command-line arguments.")
        print("To use this script, please provide:")
        print("  --run-id <your-mlflow-run-id>")
        print("  --input '<json-data>' OR --input-file <csv-path>")
        print("\nExample usage in notebook:")
        print("  %run ./predict.py --run-id abc123 --input-file test.csv")
        print("=" * 70)
        return

    if not (args.input or args.input_file):
        print("Error: You must provide either --input or --input-file")
        return

    if args.input and args.input_file:
        print("Error: Choose one: --input (single) OR --input-file (batch)")
        return

    spark = create_spark_session()

    try:
        model = load_model(args.run_id)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return

    if args.input:
        # ── Single example ───────────────────────────────────────
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
            return

        input_df = prepare_single_input(input_data, spark)
        print("\nInput record:")
        input_df.show(truncate=False)

        print("\nPrediction:")
        run_prediction(
            model,
            input_df,
            output_proba=not args.no_probability,
            output_path=None
        )

    else:
        # ── Batch from CSV ───────────────────────────────────────
        print(f"\nReading batch data from: {args.input_file}")
        input_df = prepare_batch_input(args.input_file, spark)

        output_path = args.output_file or "predictions_output"
        if not args.output_file:
            print("No --output-file given → saving to 'predictions_output' folder")

        run_prediction(
            model,
            input_df,
            output_proba=not args.no_probability,
            output_path=output_path
        )


if __name__ == "__main__":
    main()