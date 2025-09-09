from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from typing import List, Any
import mlflow
import sys
import os
import pandas as pd

#Load Model
MODEL_RUN_ID = os.environ.get("MODEL_RUN_ID")

if not MODEL_RUN_ID:
    print("MODEL_RUN_ID environment variable not set")
    sys.exit(1)

model_uri = f"runs:/{MODEL_RUN_ID}/model"

#Search mlruns folder
os.environ["MLflow_tracking_uri"] = "file:./Workflow_CI/MLProjects/mlruns"

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
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}, 500