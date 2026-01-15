import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Fungsi untuk membuang outlier (IQR) ---
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# --- Fungsi Utama Preprocessing ---
def preprocess_data(input_path, output_path):
    # 1. Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Data awal dimuat: {df.shape}")

    # 2. Handling Missing Values
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    if 'LoanAmount' in df.columns:
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    # 3. Cleaning Khusus
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', '4').astype(int)

    # 4. Encoding
    mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Loan_Status': {'Y': 1, 'N': 0}
    }
    for col, map_val in mapping.items():
        if col in df.columns:
            df[col] = df[col].map(map_val)

    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # 5. Hapus Outlier
    numeric_targets = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    existing_numeric = [c for c in numeric_targets if c in df.columns]
    
    df_clean = remove_outliers_iqr(df, existing_numeric)

    # 6. Pisahkan Fitur & Label
    target_col = 'Loan_Status'
    if target_col in df_clean.columns:
        X = df_clean.drop(columns=target_col)
        y = df_clean[target_col]
        
        # 7. Scaling & Split
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        processed_df = pd.DataFrame(X_scaled, columns=X.columns)
        processed_df[target_col] = y.values
    else:
        processed_df = df_clean

    # 8. Simpan Hasil (Langsung sebagai file, tanpa folder tambahan)
    processed_df.to_csv(output_path, index=False)
    print(f"Sukses! Data tersimpan sebagai file: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input: File 'loan_dataset.csv' di luar folder preprocessing
    input_file = os.path.join(base_dir, '..', 'loan_dataset.csv')
    
    # Output: File 'loan_dataset_preprocessing.csv' di dalam folder preprocessing
    output_file = os.path.join(base_dir, 'loan_dataset_preprocessing.csv')

    print(f"Mencari input di: {os.path.abspath(input_file)}")

    try:
        preprocess_data(input_file, output_file)
    except Exception as e:
        print(f"Terjadi Error: {e}")