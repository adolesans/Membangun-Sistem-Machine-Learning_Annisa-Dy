import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- 1. KONFIGURASI DAGSHUB---
DAGSHUB_USERNAME = "adolesans"
REPO_NAME = "Submission_SML_Annisa"           
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan!")
    return pd.read_csv(path)

def train_and_tune(df):
    print("Mempersiapkan data...")
    # Pisahkan Fitur (X) dan Target (y)
    # Pastikan nama kolom target sesuai dengan datasetmu (Loan_Status)
    target_col = 'Loan_Status' 
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 2. SETUP MLFLOW KE DAGSHUB ---
    print("Menghubungkan ke DagsHub...")
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Eksperimen_Loan_Prediction")

    # --- 3. HYPERPARAMETER TUNING (GridSearch) ---
    print("Memulai Hyperparameter Tuning...")
    
    # Kita gunakan Random Forest agar performa biasanya lebih stabil
    rf = RandomForestClassifier(random_state=42)
    
    # Parameter yang akan diujicoba
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    with mlflow.start_run():
        # Jalankan Grid Search
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)

        # Ambil model terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Parameter Terbaik: {best_params}")

        # Prediksi dengan model terbaik
        y_pred = best_model.predict(X_test)

        # Hitung Metriks
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Akurasi: {acc}")

        # --- 4. MANUAL LOGGING---
        # Log Parameter
        mlflow.log_params(best_params)
        
        # Log Metriks
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # --- 5. ARTEFAK TAMBAHAN ---
        
        # A. Artefak 1: Confusion Matrix (Gambar)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Best Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        cm_filename = "confusion_matrix.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename) 
        plt.close()

        # B. Artefak 2: Classification Report (Teks)
        report = classification_report(y_test, y_pred)
        report_filename = "classification_report.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_filename) 

        # Log Model Sklearn
        mlflow.sklearn.log_model(best_model, "model_random_forest_tuned")
        
        print("\nSUKSES! Training selesai. Cek DagsHub > Experiments.")

        # Bersihkan file lokal
        if os.path.exists(cm_filename): os.remove(cm_filename)
        if os.path.exists(report_filename): os.remove(report_filename)

if __name__ == "__main__":
    # Nama file CSV hasil preprocessing
    input_csv = "loan_dataset_preprocessing.csv"
    
    try:
        df = load_data(input_csv)
        train_and_tune(df)
    except Exception as e:
        print(f"Terjadi Eror: {e}")