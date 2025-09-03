# Ganti python_version dan mlflow_version sesuai dengan yang Anda gunakan
ARG MLFLOW_VERSION=2.18.0
FROM ghcr.io/mlflow/mlflow:v${MLFLOW_VERSION}

# 
WORKDIR /MLProjects

# file requirements.txt di direktori utama
COPY requirements.txt .

# Instal library Python
RUN pip install --no-cache-dir -r requirements.txt

# Anda perlu memastikan file-file ini ada sebelum build
COPY Preprocess_pipeline.joblib .
COPY final_model.joblib .