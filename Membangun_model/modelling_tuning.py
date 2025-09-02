from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import os
import pandas as pd
import numpy as np
import sys

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
mlflow.set_experiment("Regresi Model")

#Training Model
with mlflow.start_run():
    mlflow.autolog()
    mlflow.set_experiment("Regresi Model")

    #Hyperparameter
    parameter_grid={
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [-1, 1]
        }

    #Inisialisasi model
    lr = LinearRegression()

    #RandomSearchCV
    gridcv = GridSearchCV(estimator=lr,
                            param_grid=parameter_grid,
                            cv=5,
                            scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error'],
                            refit = 'neg_mean_absolute_error'
                        )
    #Training parameter
    gridcv.fit(X_train, y_train)

    #Parameter dan model terbaik
    best_params = gridcv.best_params_
    best_model = gridcv.best_estimator_

    #Evaluate
    y_pred = best_model.predict(X_test)

    #Metriks
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    #logging metrik 
    mlflow.log_metric("hyper_mae", mae)
    mlflow.log_metric("hyper_mse", mse)
    mlflow.log_metric("hyper_rmse", rmse)