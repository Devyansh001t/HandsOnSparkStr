"""
task4.py
--------
Task 4: Real-Time Fare Prediction Using MLlib Regression
---------------------------------------------------------
1. Offline Training: Trains a Linear Regression model on training-dataset.csv.
2. Real-Time Inference: Ingests streaming data, applies the pre-trained model,
   computes the absolute deviation between actual and predicted fares.
"""

import os
import sys

# ----------------------------
# CRITICAL WINDOWS & ENVIRONMENT PATH CORRECTIONS
# ----------------------------
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"

# Maps to your current workspace directory path structure
os.environ["SPARK_HOME"] = r"C:\Users\Devyansh Tailor\Documents\Githubfor\HandsOnSparkStr\spark-env\Lib\site-packages\pyspark"

# Forces Spark background worker/driver loops to prioritize the isolated environment 
os.environ["PYSPARK_PYTHON"] = r"C:\Users\Devyansh Tailor\Documents\Githubfor\HandsOnSparkStr\spark-env\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\Devyansh Tailor\Documents\Githubfor\HandsOnSparkStr\spark-env\Scripts\python.exe"

# Reconstruct PATH explicitly pointing to your local binaries
os.environ["PATH"] = (
    r"C:\Program Files\Java\jdk-17\bin" + os.pathsep +
    os.path.join(os.environ["SPARK_HOME"], "bin") + os.pathsep +
    r"C:\hadoop\bin" + os.pathsep +
    os.environ.get("PATH", "")
)

# Set loopback mapping variable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

# NOW WE IMPORT THE MODULES SAFELY
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, abs as abs_diff
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# TODO: Import VectorAssembler, LinearRegression, and LinearRegressionModel - COMPLETED
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# Create Spark Session with optimized local options
spark = SparkSession.builder \
    .appName("Task4_FarePrediction_Assignment") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define paths for the model and training data
MODEL_PATH = "models/fare_model"
TRAINING_DATA_PATH = "training-dataset.csv"
CHECKPOINT_DIR = "C:/tmp/checkpoints/task_4"

# --- PART 1: MODEL TRAINING (Offline) ---
# This part trains the model only if it doesn't already exist.
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] No model found. Training a new model using {TRAINING_DATA_PATH}...")

    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(f"Missing file. Please place '{TRAINING_DATA_PATH}' in your workspace folder.")

    # Load the training data from the provided CSV file
    train_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)

    # TODO: Cast `distance_km` and `fare_amount` columns to DoubleType for ML - COMPLETED
    train_df = train_df_raw \
        .withColumn("distance_km", col("distance_km").cast(DoubleType())) \
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # TODO: Create a VectorAssembler to combine feature columns into a single 'features' vector - COMPLETED
    assembler = VectorAssembler(inputCols=["distance_km"], outputCol="features")
    train_data_with_features = assembler.transform(train_df)

    # TODO: Create a LinearRegression model instance - COMPLETED
    lr = LinearRegression(featuresCol="features", labelCol="fare_amount")

    # TODO: Train the model by fitting it to the training data - COMPLETED
    model = lr.fit(train_data_with_features)

    # TODO: Save the trained model to the specified MODEL_PATH - COMPLETED
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Training Complete] Model saved to -> {MODEL_PATH}")
else:
    print(f"[Model Found] Using existing model from {MODEL_PATH}")


# --- PART 2: STREAMING INFERENCE ---
print("\n[Inference Phase] Starting real-time fare prediction stream...")

# Define the schema for the incoming streaming data
schema = StructType([
    StructField("trip_id", StringType(), True),
    StructField("driver_id", IntegerType(), True),
    StructField("distance_km", DoubleType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("timestamp", StringType(), True)
])

# Read streaming data from the socket (Binding safely to Loopback IP)
raw_stream = spark.readStream.format("socket") \
    .option("host", "127.0.0.1") \
    .option("port", 9999) \
    .load()

# Parse the incoming JSON data from the stream
parsed_stream = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*")

# TODO: Load the pre-trained LinearRegressionModel from MODEL_PATH - COMPLETED
model = LinearRegressionModel.load(MODEL_PATH)

# TODO: Use a VectorAssembler to transform the `distance_km` column of the streaming data - COMPLETED
assembler_inference = VectorAssembler(inputCols=["distance_km"], outputCol="features")
stream_with_features = assembler_inference.transform(parsed_stream)

# TODO: Use the loaded model to make predictions on the streaming data - COMPLETED
predictions = model.transform(stream_with_features)

# TODO: Calculate the 'deviation' between the actual 'fare_amount' and the 'prediction' - COMPLETED
predictions_with_deviation = predictions.withColumn(
    "deviation",
    abs_diff(col("fare_amount") - col("prediction"))
)

# Select the final columns to display in the output
output_df = predictions_with_deviation.select(
    "trip_id", "driver_id", "distance_km", "fare_amount",
    col("prediction").alias("predicted_fare"), "deviation"
)

# Write the final results to the console (Added required checkpointLocation)
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .option("checkpointLocation", CHECKPOINT_DIR) \
    .start()

# Wait for the streaming query to terminate
query.awaitTermination()