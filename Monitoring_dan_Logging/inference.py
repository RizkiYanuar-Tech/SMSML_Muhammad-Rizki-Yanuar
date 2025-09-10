from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
from typing import List, Any
import mlflow
import sys
import os
import pandas as pd

#Customize Metrik
PREDICTION_DISTRIBUTION = Histogram(
    'model_prediction_score', #Title
    'Prediction_score_distribution_Model', #Description
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
)

#Load Model
MODEL_RUN_ID = os.environ.get("MODEL_RUN_ID")

if not MODEL_RUN_ID:
    print("MODEL_RUN_ID environment variable not set")
    sys.exit(1)

model_uri = f"runs:/{MODEL_RUN_ID}/model"

#Search mlruns folder
os.environ["MLFLOW_TRACKING_URI"] = "file:/app/mlruns"

print(f"Memuat model dari URI: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
print("Load Model Successfull!")

#Data Type Definition
class PandasSplitData(BaseModel):
    columns: List[str]
    data: List[List[Any]]

class PredictionInput(BaseModel):
    dataframe_split: PandasSplitData

#Serving API Monitoring
app = FastAPI()
Instrumentator().instrument(app).expose(app)

@app.post("/invocations")
async def predict(input_data: PredictionInput):
    try:
        df = pd.DataFrame(input_data.dataframe_split.data, columns=input_data.dataframe_split.columns)
        predictions = model.predict(df)
        prediction_list = predictions.tolist()
        
        #Observe Customs Metric
        try:
            for pred_value in prediction_list:
                PREDICTION_DISTRIBUTION.observe(int(pred_value))
        except Exception as e:
            print(f"Gagal mengamati metrik: {e}")
            return {"predictions": prediction_list}
        
        #Success output
        return {"predictions": prediction_list} 
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f'Internal Server Error: {str(e)}')