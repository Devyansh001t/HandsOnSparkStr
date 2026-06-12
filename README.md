# Handson-L10-Spark-Streaming-MachineLearning-MLlib

# Real-Time Ride-Sharing Analytics & Predictive Modeling with Apache Spark


1. **Real-Time Fare Prediction and Anomaly Detection (Task 4):** Evaluates live streaming trips against an offline trained regression model to isolate price variations dynamically.
2. **Time-Based Fare Trend Prediction (Task 5):** Utilizes rolling historical aggregations and windowed feature engineering to project average regional fare movements based on cyclical time indicators.

# Project Explanations & Approach

## Task 4: Real-Time Fare Prediction Using MLlib Regression

### Approach & Methodology
* **Offline Model Training:** The pipeline reads historical baseline ride metrics from `training-dataset.csv`. The independent feature (`distance_km`) and the targeted label (`fare_amount`) columns are cleanly cast to double-precision numbers. A `VectorAssembler` transforms the raw `distance_km` column into a dense feature vector layer, which is subsequently passed to fit a `LinearRegression` model instance. Once fitted, the pipeline state is written directly to disk at `models/fare_model`.
* **Real-Time Inference:** The streaming program sets up an input stream bound via a network socket connection on port 9999. The JSON payloads are unmarshalled against a predefined structured schema using `from_json`.
* **Anomaly Detection:** The pre-trained `LinearRegressionModel` is loaded from disk and maps predictions over the live streaming vector space. An inline arithmetic transformation evaluates the absolute mathematical difference between the actual trip cost and the model's generated estimation
  
### Verified Output (Batch 95)
<img width="842" height="166" alt="Task4sample95" src="https://github.com/user-attachments/assets/a236eb5b-24bf-4702-a046-ddf2b8b6ff89" />


---

## Task 5: Time-Based Fare Trend Prediction

### Approach & Methodology
* **Feature Engineering:** Baseline historical logs from `training-dataset.csv` are aggregated into structured 5-minute event-time window frames to evaluate moving metrics and compute the baseline average fare (`avg_fare`). Instead of feeding raw timestamps directly into a model, feature engineering extracts cyclical temporal features from the window's starting boundary point: `hour_of_day` and `minute_of_hour`. 
* **Trend Estimation:** A secondary `LinearRegression` model evaluates these cyclical parameters as independent feature inputs against the windowed target variable (`avg_fare`). The computed weight distributions are exported cleanly to disk at `models/fare_trend_model_v2`.
* **Streaming Window Processing:** The active socket stream organizes records into rolling 5-minute blocks using a 1-minute sliding interval and watermark thresholds to account for late data. The micro-batches isolate the temporal indicators, feed the transformation arrays into the pre-trained trend model, and write the synchronized window boundaries alongside the predicted trend values directly to the output console terminal.

### Verified Output (Batch 16)
<img width="536" height="128" alt="task5ourputt" src="https://github.com/user-attachments/assets/30eb70e5-cade-4ff9-8ec2-a3fa8ff20459" />




