# Revolutionizing-Vehicle-Maint-Revolutionizing-Vehicle-Maintenance-with-IOT-AI-Cloud-i.mobilothon-4.0

# IoT-Enabled Predictive Maintenance System

This project provides a complete end-to-end solution for real-time vehicle component monitoring using IoT sensors, cloud storage, and machine learning to predict faults and generate alerts for preventive maintenance.

## Features
- **Real-time sensor data collection and cloud upload**: Continuously collects data from IoT sensors and uploads to an AWS S3 bucket.
- **Machine learning-based fault detection**: Applies a pre-trained model to sensor data for fault detection.
- **Automated alerts for maintenance**: Integrates AWS SNS for alert notifications when faults are detected.

## Architecture

![image](https://github.com/user-attachments/assets/98221813-3e62-4a14-a62b-4c4226ec5d4c)
### Architecture Overview
1. **IoT Sensors**: Capture real-time data (temperature, vibration, pressure, oil quality).
2. **Edge Device (e.g., Raspberry Pi)**: Collects data and uploads it to AWS S3 at regular intervals.
3. **AWS S3 Bucket**: Stores sensor data.
4. **AWS Lambda (Optional)**: Triggers data processing upon new data arrival.
5. **AWS SageMaker or Edge Inference**: Serves the machine learning model for real-time predictions.
6. **AWS SNS / Twilio**: Sends alerts to maintenance personnel.

## User Flow Diagram

1. **Sensor Data Collection**: Data is captured in real-time by IoT sensors.
2. **Data Upload**: Sensor data is sent to an AWS S3 bucket.
3. **Data Retrieval and Analysis**: Data is retrieved from S3 and processed by an ML model.
4. **Fault Detection and Alerts**: The system detects faults and sends alerts if maintenance is required.

---

## Getting Started

### Prerequisites
1. **AWS Account**: Ensure IAM permissions for S3 and SNS access.
2. **Required Libraries**:
   ```bash
   pip install boto3 pandas scikit-learn joblib
   ```
3. **Trained ML Model**: Use the provided `trained_model.pkl` or train a new model.

### Installation and Setup

1. **Set up AWS Credentials**:
   - Configure AWS CLI or add credentials to `~/.aws/credentials`:
     ```ini
     [default]
     aws_access_key_id = YOUR_AWS_ACCESS_KEY
     aws_secret_access_key = YOUR_AWS_SECRET_KEY
     ```
   
2. **Downloaded Files**:
   
[historical_sensor_data.csv](sandbox:/mnt/data/historical_sensor_data.csv): Sample data for model training and testing.
   - [trained_model.pkl](sandbox:/mnt/data/trained_model.pkl): Pre-trained model for predictions.


## Running the Code

### 1. Real-Time Data Collection and Upload to S3

This script, `data_collection.py`, collects sensor data and uploads it to S3.

```python
import boto3
import time
import random
import pandas as pd
from datetime import datetime

s3 = boto3.client('s3', aws_access_key_id='YOUR_AWS_ACCESS_KEY',
                  aws_secret_access_key='YOUR_AWS_SECRET_KEY',
                  region_name='YOUR_REGION')

bucket_name = 'your-bucket-name'
file_name = 'realtime_sensor_data.csv'

def collect_sensor_data():
    data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'temperature': random.uniform(20, 100),
        'vibration': random.uniform(0.5, 2.5),
        'pressure': random.uniform(30, 120),
        'oil_quality': random.uniform(0.7, 1.0)
    }
    return data

def upload_data_to_s3(data):
    df = pd.DataFrame([data])
    csv_data = df.to_csv(index=False, header=False)
    s3.put_object(Body=csv_data, Bucket=bucket_name, Key=file_name, ContentType='text/csv')
    print(f"Uploaded data to {bucket_name}/{file_name}")

while True:
    sensor_data = collect_sensor_data()
    upload_data_to_s3(sensor_data)
    time.sleep(5)
```

### 2. Fault Detection with Pre-Trained Model

Use `fault_detection.py` to fetch data, run predictions, and generate alerts.

```python
import boto3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

s3 = boto3.client('s3')
bucket_name = 'your-bucket-name'
file_key = 'realtime_sensor_data.csv'

def fetch_realtime_sensor_data():
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = pd.read_csv(obj['Body'], names=['timestamp', 'temperature', 'vibration', 'pressure', 'oil_quality'])
    return data

model = joblib.load("trained_model.pkl")

def predict_failure(data_row):
    prediction = model.predict([data_row])[0]
    if prediction == 0:
        return "Healthy", "No action required."
    else:
        return "Fault Detected", "Schedule maintenance."

new_data = np.array([85, 1.8, 50, 0.8])
status, solution = predict_failure(new_data)
print(f"Component Status: {status}")
print(f"Suggested Solution: {solution}")
```

### 3. Real-Time Alert System

Send alerts via AWS SNS if a fault is detected.

```python
import boto3

sns = boto3.client('sns', region_name='YOUR_REGION')

def send_alert(message, topic_arn):
    sns.publish(TopicArn=topic_arn, Message=message)
    print("Alert sent:", message)

if status == "Fault Detected":
    send_alert("Urgent: Component Fault Detected! Immediate maintenance required.", 'YOUR_SNS_TOPIC_ARN')
```

---

## Training a New Model

Use `model_training.py` to train a new model if needed.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('historical_sensor_data.csv')
X = data[['temperature', 'vibration', 'pressure', 'oil_quality']]
y = data['component_status']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, 'trained_model.pkl')
```

---

## Future Improvements
- **Scalable Model Deployment**: Deploy the model on AWS SageMaker for real-time predictions.
- **Enhanced Fault Detection**: Add more sensor inputs for comprehensive monitoring.

