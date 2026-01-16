import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import os

# --- KONFIGURASI ---
INPUT_FILE = "loan_dataset_preprocessing.csv"

def train_basic_model():
    print("=== Memulai Training Model===")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File {INPUT_FILE} tidak ditemukan di folder ini.")
    
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Split X dan y
    target_col = 'Loan_Status'
    
    if target_col not in df.columns:
        raise ValueError(f"Kolom '{target_col}' tidak ditemukan di dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow 
    mlflow.set_experiment("Basic_Training_Autolog")
    
    # ForAutolog
    # MLflow otomatis mencatat parameter, metrik, dan model tanpa kita ketik manual
    mlflow.sklearn.autolog()

    # 4. Training Model
    with mlflow.start_run():
        print("Sedang melatih model...")
        
        # Latih Random Forest standar
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi sederhana
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model berhasil dilatih! Akurasi: {acc:.4f}")
        print("Semua log (Metrics, Params, Artifacts) sudah tersimpan otomatis oleh MLflow Autolog.")

if __name__ == "__main__":
    try:
        train_basic_model()
    except Exception as e:
        print(f"Terjadi Kesalahan: {e}")
        