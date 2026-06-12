"""
task5.py
--------
Task 5: Time-Based Fare Trend Prediction
-----------------------------------------
1. Offline Training: Aggregates historical baseline data into 5-minute windows, 
   extracts hour_of_day & minute_of_hour, and trains a trend model.
2. Real-Time Inference: Applies identical 5-minute sliding window aggregations 
   and feature engineering on streaming data to project average fares.
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
from pyspark.sql.functions import from_json, col, avg, window, hour, minute
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

# TODO: Import VectorAssembler, LinearRegression, and LinearRegressionModel from pyspark.ml - COMPLETED
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# Initialize Spark Session with optimized configuration
spark = SparkSession.builder \
    .appName("Task5_FareTrendPrediction_Assignment") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Paths
MODEL_PATH = "models/fare_trend_model_v2"
TRAINING_DATA_PATH = "training-dataset.csv"
CHECKPOINT_DIR = "C:/tmp/checkpoints/task_5"

# ------------------- MODEL TRAINING (Offline) ------------------- #
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] Training new model with feature engineering using {TRAINING_DATA_PATH}...")

    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(f"Missing file. Please place '{TRAINING_DATA_PATH}' in your workspace folder.")

    # Load and process historical data
    hist_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)
    hist_df_processed = hist_df_raw.withColumn("event_time", col("timestamp").cast(TimestampType())) \
                                   .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # TODO: Aggregate data into 5-minute time windows, calculating the average fare - COMPLETED
    hist_windowed_df = hist_df_processed \
        .groupBy(window(col("event_time"), "5 minutes")) \
        .agg(avg("fare_amount").alias("avg_fare"))

    # TODO: Engineer time-based features from the window's start time - COMPLETED
    hist_features = hist_windowed_df \
        .withColumn("hour_of_day", hour(col("window.start"))) \
        .withColumn("minute_of_hour", minute(col("window.start")))

    # TODO: Create a VectorAssembler for the new time-based features - COMPLETED
    assembler = VectorAssembler(inputCols=["hour_of_day", "minute_of_hour"], outputCol="features")
    train_df = assembler.transform(hist_features)

    # TODO: Create and train the LinearRegression model - COMPLETED
    lr = LinearRegression(featuresCol="features", labelCol="avg_fare")
    model = lr.fit(train_df)

    # TODO: Save the trained model - COMPLETED
    model.write().overwrite().save(MODEL_PATH)
    print(f"[Model Saved] -> {MODEL_PATH}")
else:
    print(f"[Model Found] Using existing model at {MODEL_PATH}")


# ------------------- STREAMING INFERENCE ------------------- #
print("\n[Inference Phase] Starting real-time trend prediction stream...")

# Define the schema for incoming data
schema = StructType([
    StructField("trip_id", StringType(), True),
    StructField("driver_id", IntegerType(), True),
    StructField("distance_km", DoubleType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("timestamp", StringType(), True)
])

# Read from socket and parse data (Safely bound to local loopback)
raw_stream = spark.readStream.format("socket") \
    .option("host", "127.0.0.1") \
    .option("port", 9999) \
    .load()

parsed_stream = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*") \
    .withColumn("event_time", col("timestamp").cast(TimestampType()))

# Add a watermark to handle late-arriving data
parsed_stream = parsed_stream.withWatermark("event_time", "1 minute")

# TODO: Apply the same 5-minute windowed aggregation to the stream sliding every 1 minute - COMPLETED
windowed_df = parsed_stream \
    .groupBy(window(col("event_time"), "5 minutes", "1 minute")) \
    .agg(avg("fare_amount").alias("avg_fare"))

# TODO: Apply the same feature engineering to the streaming windowed data - COMPLETED
windowed_features = windowed_df \
    .withColumn("hour_of_day", hour(col("window.start"))) \
    .withColumn("minute_of_hour", minute(col("window.start")))

# TODO: Create a VectorAssembler for the streaming features - COMPLETED
assembler_inference = VectorAssembler(inputCols=["hour_of_day", "minute_of_hour"], outputCol="features")
feature_df = assembler_inference.transform(windowed_features)

# TODO: Load the pre-trained regression model from MODEL_PATH - COMPLETED
trend_model = LinearRegressionModel.load(MODEL_PATH)

# TODO: Use the model to make predictions on the streaming features - COMPLETED
predictions = trend_model.transform(feature_df)

# Select final columns for output
output_df = predictions.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    col("avg_fare"),
    col("prediction").alias("predicted_next_avg_fare")
)

# Write predictions to the console (Added explicit checkpoint path for reliability)
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .option("checkpointLocation", CHECKPOINT_DIR) \
    .start()

query.awaitTermination()