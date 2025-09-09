# Dockerfile
FROM python:3.12-slim
WORKDIR /app

#Receive RUN_ID from docker build
ARG MODEL_RUN_ID_ARG
ENV MODEL_RUN_ID=$MODEL_RUN_ID_ARG

#Copy path requirement-serve.txt
COPY Monitoring_dan_Logging/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy API Script
COPY Monitoring_dan_Logging/inference.py .

# Copy artifact trained model
COPY Workflow_CI/MLProjects/mlruns ./Workflow_CI/MLProjects/mlruns

#Port
EXPOSE 5000

#Run univorn server from inference app
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "5000"]