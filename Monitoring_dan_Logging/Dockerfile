# Dockerfile
FROM python:3.12-slim
WORKDIR /app

#Receive RUN_ID from docker build
ARG MODEL_RUN_ID_ARG
ENV MODEL_RUN_ID=$MODEL_RUN_ID_ARG

#Copy path requirement-serve.txt
COPY Monitoring_dan_Logging/requirement-serve.txt .
RUN pip install --no-cache-dir -r requirement-serve.txt

# Copy API Script
COPY Monitoring_dan_Logging/inference.py .

# Copy artifact trained model
COPY mlruns ./mlruns

# Port 8000
EXPOSE 8000

# Gunicorn as manager and uvicorn as worker
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "inference:app", "-b", "0.0.0.0:8000"]