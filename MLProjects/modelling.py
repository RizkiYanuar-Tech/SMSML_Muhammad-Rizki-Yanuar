import mlflow
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

#Get path this directory
current_dir = os.path.dirname(os.path.abspath(__file__))

#Get root project
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Project root ditambahkan ke path: {project_root}")

from Preprocessing.Automate_MuhammadRizkiYanuar import preprocess

#LOAD DATA
data_path = os.path.join(project_root, 'StudentPerformanceFactors.csv')

try:
    stuper = pd.read_csv(data_path)
    print(f"\nDataset berhasil dimuat dari: {data_path}")
except FileNotFoundError:
    print(f"file tidak ditemukan di {data_path}")

#Preprocessing & Splitting Data
X_train, X_test, y_train, y_test = preprocess(stuper, 'Exam_Score', ['Hours_Studied','Tutoring_Sessions'], 'Preprocess_pipeline.joblib', 'Train_studentperformancefactors.csv', 'Test_studentperformancefactors.csv')

#Mlflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
#Experiment Title
mlflow.set_experiment("Model Final")

#Training Model
with mlflow.start_run():
    mlflow.autolog()
    #Inisialisasi model
    lr = LinearRegression()

    #Training
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    #Metrik evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    #Lacak log metrik
    mlflow.log_metric("mae",mae)
    mlflow.log_metric("mse",mse)
    mlflow.log_metric("rmse",rmse)