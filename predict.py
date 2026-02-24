
#Batch processing
import pandas as pd
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pyspark.sql import Row
from pyspark.sql.functions import col,when,lit,sum,round, expr,mean, stddev

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

data = pd.read_csv('bank_marketing_data.csv',sep=';')
data =data.drop('Unnamed: 0', axis=1)
data =data.drop('target', axis=1)


def data_preprocessing(data):
    data2 = spark.createDataFrame(data)#.drop('Unnamed: 0')

    dataColumns = data.columns

    df_unique = spark.createDataFrame([Row(column=c, unique_count=data2.select(c).dropDuplicates().count()) for c in dataColumns])

    #Split into categorical and contionous columns using the unique counts
    contionousColumns =  df_unique.filter(col('unique_count')>32)
    categoricalColumns =  df_unique.filter(col('unique_count')<=32)
    categoricalCol = categoricalColumns.select('column').rdd.flatMap(lambda x: x).collect()

    data = pd.get_dummies(
        data,
        drop_first=True,          
        columns=[i for i in categoricalCol if 'target' not in i],  # only convert these columns
        dtype=int                
    )


    #impute missing values
    imputer = KNNImputer(
        n_neighbors=2,
        weights="uniform",     
        add_indicator=False    
    )

    numeric_cols = [i for i in data.columns]

    df_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(data[numeric_cols]),
        columns=numeric_cols,
        index=data.index
    )


    #Normalize your dataset
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_Normalized = pd.DataFrame(scaler.fit_transform(df_numeric_imputed), columns=df_numeric_imputed.columns)
    
    return data_Normalized
    
data = data_preprocessing(data)
model_uri = 'runs:/a7b04f98a6ca4a2087b1fe0e3aea4ea4/model'
model = mlflow.pyfunc.load_model(model_uri)

from pyspark.ml.feature import VectorAssembler

# Convert to Spark DataFrame and assemble features into vector
data_spark = spark.createDataFrame(data)
assembler = VectorAssembler(inputCols=data.columns.tolist(), outputCol='features')
data_with_features = assembler.transform(data_spark).select('features')

# Predict on the assembled features
model.predict(data_with_features.toPandas()['features'].apply(lambda x: x.toArray().tolist()))