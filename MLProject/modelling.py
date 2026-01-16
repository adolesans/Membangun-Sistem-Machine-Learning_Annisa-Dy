import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

INPUT_FILE = "loan_dataset_preprocessing.csv"

def train():
    print("=== Memulai Training Otomatis... ===")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} tidak ditemukan!")
        return
    
    # Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        X = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        print("Sedang melatih model...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"model berhasil dilatih. Akurasi: {acc:.4f}")
        
    except Exception as e:
        print(f"Terjadi Error saat training: {e}")

if __name__ == "__main__":
    train()