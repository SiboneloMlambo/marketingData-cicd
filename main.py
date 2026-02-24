# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import numpy as np
from pyspark.sql import Row
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import os
# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting customer subscription to a term deposit based on marketing campaign data.",
    version="1.0.0"
)
# --- Global Variables for Model and Spark Session ---
# This ensures SparkSession and MLflow model are loaded only once at startup
spark: SparkSession = None
loaded_model = None
feature_columns = None # To store the ordered list of feature columns after preprocessing
data_preprocessing_instance = None # Store a reference to the preprocessor, if needed
# --- Configuration ---
# Assuming 'bank_marketing_data.csv' is in the same directory as main.py
DATA_FILE_PATH = 'bank_marketing_data.csv'
MODEL_URI = 'runs:/a7b04f98a6ca4a2087b1fe0e3aea4ea4/model' 
# --- Pydantic Model for Input Data ---
# Define the structure of your input data based on the bank_marketing_data.csv
# This should match the columns *before* any preprocessing (except 'Unnamed: 0' and 'target')
class BankMarketingData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
# --- Data Preprocessing Function (Modified for API) ---
class DataPreprocessor:
    def __init__(self, spark_session, training_data_path):
        self.spark = spark_session
        self.categorical_cols_to_onehot = []
        self.numerical_cols_for_imputation = []
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=2, weights="uniform", add_indicator=False)
        self.all_processed_columns = [] # To store the final column order after one-hot encoding
        # Load and fit preprocessors on initial data
        self._fit_preprocessors(training_data_path)
    def _fit_preprocessors(self, training_data_path):
        # This function loads a sample of the training data to fit the preprocessors
        # so that subsequent new data can be transformed consistently.
        print("Fitting data preprocessors on training data...")
        raw_training_data = pd.read_csv(training_data_path, sep=';')
        raw_training_data = raw_training_data.drop('Unnamed: 0', axis=1, errors='ignore')
        raw_training_data = raw_training_data.drop('target', axis=1, errors='ignore')
        # Identify categorical columns from the training data
        training_data_spark = self.spark.createDataFrame(raw_training_data)
        df_unique = self.spark.createDataFrame([
            Row(column=c, unique_count=training_data_spark.select(c).dropDuplicates().count())
            for c in raw_training_data.columns
        ])
        categorical_columns_df = df_unique.filter(col('unique_count') <= 32)
        self.categorical_cols_to_onehot = [
            i for i in categorical_columns_df.select('column').rdd.flatMap(lambda x: x).collect()
            if 'target' not in i
        ]
        # Apply one-hot encoding to training data to determine all possible columns
        encoded_training_data = pd.get_dummies(
            raw_training_data,
            drop_first=True,
            columns=self.categorical_cols_to_onehot,
            dtype=int
        )
        self.numerical_cols_for_imputation = list(encoded_training_data.columns)
        self.all_processed_columns = list(encoded_training_data.columns)
        # Fit imputer and scaler
        self.imputer.fit(encoded_training_data[self.numerical_cols_for_imputation])
        self.scaler.fit(self.imputer.transform(encoded_training_data[self.numerical_cols_for_imputation]))
        print("Data preprocessors fitted successfully.")
    def transform(self, data_to_process: pd.DataFrame) -> pd.DataFrame:
        print("Transforming new data...")
        # Ensure new data has same columns as training data before encoding
        # This handles cases where a new prediction doesn't have all categories
        for col_name in self.categorical_cols_to_onehot:
            if col_name not in data_to_process.columns:
                data_to_process[col_name] = np.nan # Or a sensible default
        # Apply one-hot encoding
        encoded_data = pd.get_dummies(
            data_to_process,
            drop_first=True,
            columns=self.categorical_cols_to_onehot,
            dtype=int
        )
        # Reindex to ensure all columns from training are present, filling missing with 0
        # This is crucial for consistent feature vectors
        missing_cols = set(self.all_processed_columns) - set(encoded_data.columns)
        for c in missing_cols:
            encoded_data[c] = 0
        encoded_data = encoded_data[self.all_processed_columns] # Ensure column order is consistent
        # Impute missing values (if any, though get_dummies should handle most)
        df_numeric_imputed = pd.DataFrame(
            self.imputer.transform(encoded_data[self.numerical_cols_for_imputation]),
            columns=self.numerical_cols_for_imputation,
            index=encoded_data.index
        )
        # Normalize dataset
        data_normalized = pd.DataFrame(
            self.scaler.transform(df_numeric_imputed),
            columns=df_numeric_imputed.columns
        )
        print("Data transformation complete.")
        return data_normalized
# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    global spark, loaded_model, feature_columns, data_preprocessing_instance
    print("Initializing SparkSession...")
    spark = SparkSession.builder \
       .appName("MLflow Spark Prediction API") \
       .getOrCreate()
    print("SparkSession initialized.")
    print(f"Loading MLflow model from {MODEL_URI}...")
    try:
        loaded_model = mlflow.pyfunc.load_model(MODEL_URI)
        print("MLflow model loaded successfully.")
    except Exception as e:
        print(f"Error loading MLflow model: {e}")
        # Optionally raise an exception to prevent app startup if model is crucial
        raise RuntimeError(f"Could not load MLflow model: {e}")
    # Initialize and fit the preprocessor
    data_preprocessing_instance = DataPreprocessor(spark, DATA_FILE_PATH)
    feature_columns = data_preprocessing_instance.all_processed_columns
# --- FastAPI Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    global spark
    if spark:
        print("Stopping SparkSession...")
        spark.stop()
        print("SparkSession stopped.")
# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(data: BankMarketingData):
    if loaded_model is None or spark is None or data_preprocessing_instance is None:
        raise HTTPException(status_code=503, detail="Service not ready: Model or Spark not loaded.")
    try:
        # Convert Pydantic model to Pandas DataFrame
        input_df = pd.DataFrame([data.dict()])
        # Preprocess the input data
        processed_data = data_preprocessing_instance.transform(input_df)
        # Convert to Spark DataFrame and assemble features into vector
        data_spark = spark.createDataFrame(processed_data)
        assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
        data_with_features = assembler.transform(data_spark).select('features')
        # Predict using the MLflow model
        # The MLflow pyfunc model expects a pandas DataFrame with features as lists
        predictions = loaded_model.predict(
            data_with_features.toPandas()['features'].apply(lambda x: x.toArray().tolist())
        )
        # Assuming your model returns a single prediction for a single input
        return {"prediction": predictions.tolist()}
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prediction: {str(e)}")
# --- Root Endpoint (Optional, for health check or info) ---
@app.get("/")
async def root():
    return {"message": "Bank Marketing Prediction API. Visit /docs for API documentation."}