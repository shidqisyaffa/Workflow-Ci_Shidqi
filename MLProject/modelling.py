"""
modelling.py - MLflow Model Training for CI/CD Workflow
========================================================
File ini melatih model Machine Learning menggunakan MLflow autolog.
Dataset: gpu_data_processed.csv (preprocessed GPU dataset)

Level: SKILLED (3 poin)
- Menggunakan MLflow autolog untuk mencatat parameter, metrik, dan model
- Compatible dengan GitHub Actions CI workflow
- Model artifacts disimpan untuk upload ke GitHub Artifacts
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# ============================================
# 1. Konfigurasi MLflow Tracking
# ============================================
# Gunakan environment variable jika tersedia, untuk CI compatibility
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using MLflow Tracking URI: {tracking_uri}")
else:
    # Default: local mlruns folder
    mlflow.set_tracking_uri("file:./mlruns")
    print("Using local mlruns folder for tracking")

mlflow.set_experiment("GPU_Classification_CI")

# ============================================
# 2. Load Dataset Preprocessing
# ============================================
print("Loading preprocessed dataset...")
# Support running from different directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "gpu_data_processed.csv")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================
# 3. Menentukan Fitur dan Target
# ============================================
# Target: manufacturer_encoded (klasifikasi manufacturer GPU)
# Fitur: semua kolom numerik kecuali target dan productName

target_column = "manufacturer_encoded"
exclude_columns = ["productName", target_column]

feature_columns = [col for col in df.columns if col not in exclude_columns]
print(f"\nTarget: {target_column}")
print(f"Features ({len(feature_columns)}): {feature_columns}")

X = df[feature_columns]
y = df[target_column]

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Unique classes: {y.nunique()}")

# ============================================
# 4. Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# ============================================
# 5. MLflow Autolog dan Training Model
# ============================================
print("\n" + "="*50)
print("Starting MLflow Autolog Training...")
print("="*50)

# Aktifkan MLflow autolog untuk sklearn
mlflow.sklearn.autolog(log_models=True)

# Training model dengan autolog
with mlflow.start_run(run_name="RandomForest_CI_Training"):
    # Buat dan latih model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training RandomForestClassifier...")
    model.fit(X_train, y_train)
    
    # Evaluasi
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Log tambahan manual
    mlflow.log_param("dataset_name", "gpu_data_processed.csv")
    mlflow.log_param("target_column", target_column)
    mlflow.log_param("n_features", len(feature_columns))
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])
    
    # Log run info
    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow Run ID: {run_id}")
    
    print("\n" + "="*50)
    print("MLflow Autolog Training Complete!")
    print("="*50)
    print("\nSemua parameter, metrik, dan model telah disimpan ke MLflow.")
    print(f"Artifacts tersimpan di folder mlruns/")
