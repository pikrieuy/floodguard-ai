# 🌊 FloodGuard: Ciliwung Early Warning & Decision Support System

> **Tugas Akhir Assignment of Learning (AoL) - Artificial Intelligence (COMP6468031 - Kelas LB75)**  
> **School of Computer Science, BINUS University @Bandung (Semester Ganjil 2026/2027)**  
> **Dosen Pengampu:** Dr. Ir. Selvi Lukman, S.T., M.T., IPM, ASEAN Eng. (D6947)

---

## 👥 Tim Peneliti (Kelompok 2)
* **Kevin Tanardi** (NIM: 2902574630) — *Machine Learning & Data Preprocessing*
* **Marcello Aquino Tirta** (NIM: 2902696842) — *Studi Literatur, Etika AI, & Laporan Ilmiah*
* **Usman Amanul Vickry** (NIM: 2902690510) — *Aplikasi MVP, Deployment, & Presenter*

---

## 🌐 Tautan Penting
* **Live Demo Web (HTTPS):** [https://beemate.web.id/floodguard/](https://beemate.web.id/floodguard/)
* **Alternatif Server IP:** [http://43.134.118.130/floodguard/](http://43.134.118.130/floodguard/)
* **Laporan Proyek 16 Bab (Google Docs):** [Dokumen Laporan Akhir PKM-KC](https://docs.google.com/document/d/1VtCjWk9uuVKDkRMmcYeflXQ2wGN8nsKlRHlkFtXMsSI/edit)
* **Slide Presentasi (Google Slides):** [Slide Presentasi 12 Slide 16:9](https://docs.google.com/presentation/d/1BFodItXBIQyANYn1konhlrGSVNH9ot6uRuYDX8qNaas/edit)

---

## 🎯 Ringkasan Proyek
FloodGuard adalah sistem pendukung keputusan (*Decision Support System*) berbasis kecerdasan buatan (*Machine Learning*) untuk memprediksi status siaga Pintu Air Manggarai (Sungai Ciliwung, DKI Jakarta).

Sistem memanfaatkan karakteristik hidrologi berupa **jeda waktu rambat air (lag time) 6 hingga 9 jam** dari Bendung Katulampa (Bogor) menuju Manggarai (Jakarta Pusat), sehingga peringatan dini dan protokol pembukaan pintu air dapat dieksekusi 3 sampai 6 jam sebelum banjir bandang meluap ke pemukiman warga.

### 🏆 Capaian Model Machine Learning
* **Algoritma Utama:** Random Forest Classifier (100 Decision Trees, Max Depth 8).
* **Ukuran Dataset:** 1.500 sampel pengamatan deret waktu resmi BPBD DKI Jakarta & BMKG.
* **Akurasi Validasi:** **96,67%** (Standar minimal rubrik BINUS: 85,0%).
* **Precision Makro:** **95,67%** (Meminimalkan alarm palsu).
* **AI Safety:** **0% False Negative pada Status Kritis** (100% potensi banjir besar terdeteksi tepat waktu).
* **Komparasi Baseline:**
  * *Logistic Regression:* Test Acc 97,00% (kaku terhadap batas non-linear cuaca).
  * *Single Decision Tree:* Test Acc 95,67% (variansi tinggi, rentan overfitting).
  * *Random Forest (Usulan):* Test Acc **96,67%** (5-Fold CV: 95,73% ± 0,57%).

---

## 🚀 Fitur Unggulan Sistem
1. **Desain Antarmuka Dribbble Bento Grid:** Tampilan modern dengan palet warna pastel, radius kartu 28px, dan aset visual cuaca 3D *claymorphic*.
2. **Explainable AI (XAI) TreeSHAP Decomposition:** Membedah kontribusi pasti kenaikan sentimeter tiap posko (Katulampa 47,3%, Depok 38,8%, Hujan Lokal 13,9%) untuk membuktikan model bukan *black-box*.
3. **Countdown Lead-Time & Jendela Evakuasi:** Estimasi waktu tiba puncak gelombang banjir dan sisa waktu evakuasi aman bagi warga bantaran kali.
4. **Peta Spasial GIS & Buffer Zona Genangan (Leaflet):** Peta otomatis fokus ke Jakarta saat banjir kritis, menampilkan radius genangan di Kampung Melayu, Bidara Cina, dan Bukit Duri.
5. **Simulator Siaran Darurat BPBD:** Simulasi pengiriman notifikasi SMS LBA dan WhatsApp Gateway resmi dengan teks arahan evakuasi otomatis.
6. **Simulasi Sensor Fault-Tolerant:** Kemampuan merekonstruksi nilai hulu secara otomatis via regresi linier ketika sensor Katulampa mati/hanyut.

---

## 📂 Struktur Repositori
```text
floodguard-ai/
├── app.py                         # Aplikasi web Streamlit (visualisasi & telemetri)
├── server.py                      # Backend API FastAPI & Uvicorn
├── train_model.py                 # Skrip pelatihan model ML & uji 5-fold CV
├── ciliwung_flood_data.csv        # Dataset hidrologi 1.500 sampel BPBD/BMKG
├── model_floodguard.joblib        # Biner model Random Forest terlatih
├── model_comparison_metrics.json # Metrik komparasi ilmiah 3 model
├── requirements.txt               # Daftar dependensi Python
├── .gitignore
├── static/
│   ├── index.html                 # Frontend Dribbble Bento Grid UI
│   └── icons/                     # Aset ikon 3D cuaca claymorphic
└── README.md
```

---

## 💻 Panduan Menjalankan Lokal

### 1. Kloning Repositori & Persiapan Lingkungan
```bash
git clone https://github.com/pikrieuy/floodguard-ai.git
cd floodguard-ai

# Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # Di Windows: .venv\Scripts\activate

# Pasang dependensi
pip install -r requirements.txt
```

### 2. Melatih Ulang Model (Opsional)
```bash
python train_model.py
```

### 3. Menjalankan Dashboard Web Dribbble UI
```bash
uvicorn server:app --host 0.0.0.0 --port 8501 --reload
```
Buka peramban di `http://localhost:8501`.

### 4. Menjalankan Versi Alternatif Streamlit
```bash
streamlit run app.py
```

---

## 📜 Lisensi & Atribusi
Proyek ini dikembangkan untuk tujuan akademis mata kuliah Artificial Intelligence, BINUS University @Bandung. Data bersumber dari Portal Open Data DKI Jakarta dan BMKG.
