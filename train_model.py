import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def run_comprehensive_training():
    np.random.seed(42)
    n_samples = 1500

    # 1. Parameter Hidrologi Daerah Aliran Sungai (DAS) Ciliwung
    curah_hujan_hulu = np.random.exponential(scale=20.0, size=n_samples)
    
    # Ketinggian Muka Air (TMA) Katulampa (cm): Normal 20-50, Waspada 80-150, Kritis >150
    katulampa = 35.0 + (curah_hujan_hulu * 2.2) + np.random.normal(0, 8.0, n_samples)
    katulampa = np.clip(katulampa, 20.0, 260.0)

    # Pos Pantau Depok (cm): jeda rambat ~3 jam dari Katulampa
    depok = 70.0 + (katulampa * 0.92) + np.random.normal(0, 10.0, n_samples)
    depok = np.clip(depok, 40.0, 390.0)

    # Hujan lokal Jakarta (mm) dan durasi (jam)
    curah_hujan_jkt = np.random.exponential(scale=15.0, size=n_samples)
    durasi_hujan = np.random.uniform(0.5, 8.0, size=n_samples)

    # Elevasi Pintu Air Manggarai (cm): kumulatif hulu + tengah + lokal
    elevasi_manggarai = 440.0 + (katulampa * 1.65) + (depok * 0.78) + (curah_hujan_jkt * 1.45) + np.random.normal(0, 12.0, n_samples)

    # Status Siaga: 0 = Normal (<750 cm), 1 = Waspada (750-850 cm), 2 = Kritis (>850 cm)
    status_siaga = np.zeros(n_samples, dtype=int)
    status_siaga[elevasi_manggarai >= 750.0] = 1
    status_siaga[elevasi_manggarai >= 850.0] = 2

    # Buat DataFrame
    df = pd.DataFrame({
        'katulampa_lag6': np.round(katulampa, 1),
        'depok_lag3': np.round(depok, 1),
        'curah_hujan_jkt': np.round(curah_hujan_jkt, 1),
        'durasi_hujan': np.round(durasi_hujan, 1),
        'elevasi_manggarai': np.round(elevasi_manggarai, 1),
        'status_siaga': status_siaga
    })

    csv_path = "/root/projects/floodguard/ciliwung_flood_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset berhasil disimpan di: {csv_path} ({len(df)} baris)")

    # 2. Pembagian Data 80% Train : 20% Test
    X = df[['katulampa_lag6', 'depok_lag3', 'curah_hujan_jkt', 'durasi_hujan']]
    y = df['status_siaga']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Model 1: Baseline Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    cv_scores_lr = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')

    # 4. Model 2: Baseline Single Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    cv_scores_dt = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')

    # 5. Model 3: Proposed Random Forest (100 Trees)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

    # Evaluasi Metrik
    results = {
        "models": {
            "Logistic Regression": {
                "test_accuracy": float(accuracy_score(y_test, y_pred_lr)),
                "cv_accuracy_mean": float(np.mean(cv_scores_lr)),
                "cv_accuracy_std": float(np.std(cv_scores_lr)),
                "precision_macro": float(precision_score(y_test, y_pred_lr, average='macro')),
                "recall_macro": float(recall_score(y_test, y_pred_lr, average='macro')),
                "f1_macro": float(f1_score(y_test, y_pred_lr, average='macro')),
                "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist()
            },
            "Single Decision Tree": {
                "test_accuracy": float(accuracy_score(y_test, y_pred_dt)),
                "cv_accuracy_mean": float(np.mean(cv_scores_dt)),
                "cv_accuracy_std": float(np.std(cv_scores_dt)),
                "precision_macro": float(precision_score(y_test, y_pred_dt, average='macro')),
                "recall_macro": float(recall_score(y_test, y_pred_dt, average='macro')),
                "f1_macro": float(f1_score(y_test, y_pred_dt, average='macro')),
                "confusion_matrix": confusion_matrix(y_test, y_pred_dt).tolist()
            },
            "Random Forest (Proposed)": {
                "test_accuracy": float(accuracy_score(y_test, y_pred_rf)),
                "cv_accuracy_mean": float(np.mean(cv_scores_rf)),
                "cv_accuracy_std": float(np.std(cv_scores_rf)),
                "precision_macro": float(precision_score(y_test, y_pred_rf, average='macro')),
                "recall_macro": float(recall_score(y_test, y_pred_rf, average='macro')),
                "f1_macro": float(f1_score(y_test, y_pred_rf, average='macro')),
                "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
                "feature_importance": {col: float(imp) for col, imp in zip(X.columns, rf.feature_importances_)}
            }
        }
    }

    # Simpan Hasil Komparasi ke JSON
    metrics_path = "/root/projects/floodguard/model_comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Hasil komparasi tersimpan di: {metrics_path}")

    # Simpan Model Terlatih
    model_path = "/root/projects/floodguard/model_floodguard.joblib"
    joblib.dump(rf, model_path)
    print(f"Model Random Forest tersimpan di: {model_path}")

    print("\n=== RINGKASAN KOMPARASI MODEL ===")
    for m_name, m_data in results["models"].items():
        print(f"[{m_name}]")
        print(f"  Test Accuracy : {m_data['test_accuracy']*100:.2f}%")
        print(f"  5-Fold CV Acc : {m_data['cv_accuracy_mean']*100:.2f}% (±{m_data['cv_accuracy_std']*100:.2f}%)")
        print(f"  Macro F1-Score: {m_data['f1_macro']*100:.2f}%")
        print(f"  Confusion Matrix:\n  {m_data['confusion_matrix']}")

if __name__ == "__main__":
    run_comprehensive_training()
