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

repo_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Project root ditambahkan ke path: {project_root}")

if repo_root not in sys.path:
    sys.path.append(repo_root)

from Preprocessing.Automate_MuhammadRizkiYanuar import preprocess

#LOAD DATA
data_path = os.path.join(repo_root, 'StudentPerformanceFactors.csv')

try:
    stuper = pd.read_csv(data_path)
    print(f"\nDataset berhasil dimuat dari: {data_path}")
except FileNotFoundError:
    print(f"file tidak ditemukan di {data_path}")
    sys.exit(1)

#Preprocessing & Splitting Data
X_train, X_test, y_train, y_test = preprocess(stuper, 'Exam_Score', ['Hours_Studied','Tutoring_Sessions'], 'Preprocess_pipeline.joblib', 'Train_studentperformancefactors.csv', 'Test_studentperformancefactors.csv')

#Training Model
mlflow.autolog(log_models=False)
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

#Simpan log model
print("Simpan model")
mlflow.sklearn.log_model(lr, "model")