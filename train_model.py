import pyspark
import pandas as pd
import seaborn as sns
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col,when,lit,sum,round, expr,mean, stddev
import matplotlib.pyplot as plt
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler,OneHotEncoderModel, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier,LogisticRegression,MultilayerPerceptronClassifier
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType,DoubleType
import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, regexp_replace,split
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from sklearn.impute import KNNImputer
import mlflow
import mlflow.spark   # important for pyspark.ml models
from mlflow.models import infer_signature
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


data_spark = spark.createDataFrame(pd.read_csv('bank_marketing_data_normalized.csv')).drop(*['Unnamed: 0.1','Unnamed: 0'])

# Prepare features: all columns except 'target'
feature_cols = [col for col in data_spark.columns if col != 'target']

# Assemble features into a vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data_assembled = assembler.transform(data_spark)

# Rename 'target' to 'label' for PySpark ML
data_ml = data_assembled.withColumnRenamed('target', 'label')
train_data, test_data = data_ml.randomSplit([0.80, 0.2], seed=1234)

#Train the model with the data as is for now
mlflow.pyspark.ml.autolog(log_models=True)
"""
with mlflow.start_run():
    lrModel = LogisticRegression(maxIter=10, regParam=0.01).fit(train_data)
"""
with mlflow.start_run(run_name="LogisticRegression-baseline") as run:
    
    # Enable auto-logging (models, metrics, params, ...)
    mlflow.autolog(log_models=True, log_input_examples=True)
    
    # Fit
    model = LogisticRegression(maxIter=10, regParam=0.01).fit(train_data)
    
    # ── 7. Predict & Evaluate ───────────────────────────────────────
    predictions = model.transform(test_data)
    
    # AUC (most important for imbalanced classification)
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )
    auc = evaluator_auc.evaluate(predictions)
    
    # Accuracy, precision, recall, f1
    evaluator_multi = MulticlassClassificationEvaluator(labelCol="label")
    accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
    f1       = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
    
    # Log extra/custom metrics
    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1", f1)
    
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1-score  : {f1:.4f}")
    
    # Optional: log test set size, class balance, etc.
    mlflow.log_param("train_count", train_data.count())
    mlflow.log_param("test_count",  test_data.count())

print(f"Run ID: {run.info.run_id}")
