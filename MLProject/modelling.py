"""
modelling.py - Final Model Training for Workflow CI
Menggunakan parameter terbaik dari hasil tuning (Kriteria 2).
Author: Fara Katty Sabila
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import os
import shutil
import joblib
import warnings

warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """Load preprocessed data dari folder diabetes_preprocessing"""
    try:
        X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
        X_test = pd.read_csv('diabetes_preprocessing/X_test.csv')
        y_train = pd.read_csv('diabetes_preprocessing/y_train.csv')
        y_test = pd.read_csv('diabetes_preprocessing/y_test.csv')
        
        print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
        
    except FileNotFoundError as e:
        print(f"Error: Dataset tidak ditemukan! Pastikan folder 'diabetes_preprocessing' ada. {e}")
        raise

def train_final_model():
    """Train model dengan BEST PARAMETERS hasil tuning"""
    
    # Nama eksperimen khusus untuk tracking otomatis di DagsHub
    mlflow.set_experiment("Diabetes_Workflow_CI")
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Aktifkan autolog agar metrik (accuracy, dll) tercatat otomatis
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Final_BestParams", nested=True):
        print("\n=== TRAINING BEST MODEL VIA WORKFLOW CI ===")
        
        # Parameter dari hasil tuning (Kriteria 2)
        model = RandomForestClassifier(
            n_estimators=500,           
            max_depth=20,               
            criterion='gini',         
            min_samples_split=2,       
            min_samples_leaf=1,        
            max_features='sqrt',        
            bootstrap=False,            
            class_weight='balanced',    
            random_state=42,           
            n_jobs=-1
        )
        
        print("Training RandomForest dengan Best Parameters...")
        model.fit(X_train, y_train)
        
        
        # Simpan model secara eksplisit ke MLflow Tracking (DagsHub)
        mlflow.sklearn.log_model(model, "model")

        # SAVE LOCALLY (Untuk memudahkan Build Docker di GitHub Actions)
        local_model_path = "online_model"
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
            
        mlflow.sklearn.save_model(
            sk_model=model,
            path=local_model_path,
            input_example=X_train.iloc[:5]
        )
        
        # Simpan model secara lokal untuk artefak CI
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model_final.pkl")
        
        print("Model training complete and logged to MLflow.")
        return model

if __name__ == "__main__":
    print("=== STARTING WORKFLOW CI TRAINING SESSION ===")
    train_final_model()