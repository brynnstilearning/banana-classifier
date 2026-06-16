# 🍌 Banana Ripeness Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10.14-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn&logoColor=white)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple?logo=railway&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Sistem klasifikasi kematangan pisang berbasis Machine Learning menggunakan fitur warna dan tekstur**

[🌐 Live Demo](https://banana-classifire-103.up.railway.app) • [📓 Notebook](datascience_UAS_banana.ipynb) • [📄 Laporan](#)

</div>

---

## 📌 Deskripsi

Project ini membangun sistem klasifikasi otomatis untuk menentukan tingkat kematangan pisang berdasarkan analisis citra digital. Sistem menggunakan **80 fitur** gabungan dari ruang warna RGB, HSV, dan pola tekstur LBP yang diekstrak dari gambar pisang, kemudian diklasifikasikan menggunakan tiga algoritma Machine Learning.

Sistem ini diimplementasikan sebagai **aplikasi web** yang dapat diakses secara publik, memungkinkan pengguna untuk mengunggah gambar pisang dan mendapatkan hasil prediksi secara real-time.
  
> 👨‍💻 **Nur Muhammad Anang Febriananto ** (230605110103)  
> 🏛️ Teknik Informatika — UIN Maulana Malik Ibrahim Malang  

---

## 🎯 Kelas yang Diklasifikasikan

| Kelas | Label | Deskripsi |
|-------|-------|-----------|
| 🟢 Mentah | `Banana_Mentah` | Kulit berwarna hijau, pisang belum matang |
| 🍌 Matang | `Banana_Matang` | Kulit berwarna kuning, siap dikonsumsi |
| 🟤 Terlalu Matang | `Banana_Terlalu_Matang` | Kulit kecoklatan, tekstur terlalu lembek |

---

## 🚀 Live Demo

Akses aplikasi web langsung di:

**[https://banana-classifire-103.up.railway.app](https://banana-classifire-103.up.railway.app)**

### Cara Penggunaan:
1. Buka URL di atas melalui browser
2. Upload gambar pisang (JPG/JPEG/PNG)
3. Pilih model Machine Learning (KNN, SVM, atau Random Forest)
4. Klik **Prediksi Kematangan**
5. Lihat hasil prediksi, confidence, dan probabilitas per kelas

---

## 📊 Hasil Evaluasi Model

| Model | Akurasi | Presisi | Recall | Specificity | F1-Measure |
|-------|---------|---------|--------|-------------|------------|
| KNN (K=2) | 96.67% | 96.72% | 96.67% | 98.33% | 96.67% |
| SVM (RBF) | 97.33% | 97.35% | 97.33% | 98.67% | 97.33% |
| **Random Forest** | **98.00%** | **98.03%** | **98.00%** | **99.00%** | **98.01%** |

### 📈 McNemar's Test (α = 0.05)

| Perbandingan | p-value | Kesimpulan |
|---|---|---|
| KNN vs SVM | 1.0000 | Tidak berbeda signifikan |
| KNN vs Random Forest | 0.5000 | Tidak berbeda signifikan |
| SVM vs Random Forest | 1.0000 | Tidak berbeda signifikan |

> ✅ Ketiga model tidak memiliki perbedaan signifikan secara statistik, namun **Random Forest dipilih** sebagai model terbaik berdasarkan akurasi numerik tertinggi.

---

## 🧠 Fitur yang Diekstrak (80 Fitur)

```
Gambar Input (64×64 px)
        │
        ├── Mean & Std RGB      →   6 fitur
        │   (R, G, B — mean & std)
        │
        ├── Histogram HSV       →  48 fitur
        │   (16 bin × 3 kanal: H, S, V)
        │
        └── Histogram LBP       →  26 fitur
            (P=8, R=1, uniform)
                                   ─────────
                          Total :  80 fitur
```

---

## 🗂️ Struktur Project

```
banana-classifier/
│
├── models/                     # Model Machine Learning (pkl)
│   ├── model_rf.pkl            # Random Forest
│   ├── model_svm.pkl           # Support Vector Machine
│   ├── model_knn.pkl           # K-Nearest Neighbor
│   ├── scaler.pkl              # StandardScaler
│   ├── label_encoder.pkl       # LabelEncoder
│   └── best_k.pkl              # Nilai K terbaik KNN
│
├── templates/
│   └── index.html              # Tampilan antarmuka web (dark mode)
│
├── static/                     # Asset statis (CSS, gambar)
│
├── app.py                      # Flask server & endpoint prediksi
├── requirements.txt            # Dependensi Python
├── Procfile                    # Konfigurasi Railway deployment
├── runtime.txt                 # Versi Python (3.10.14)
├── SAVE_MODEL_CELL.py          # Script untuk menyimpan model dari Colab
└── README.md
```

---

## ⚙️ Instalasi & Menjalankan Lokal

### Prasyarat
- Python 3.10.14
- File model `.pkl` (download dari Google Colab setelah training)

### Langkah-langkah

```bash
# 1. Clone repository
git clone https://github.com/brynnstilearning/banana-classifier.git
cd banana-classifier

# 2. Buat virtual environment
python -m venv venv

# 3. Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependensi
pip install -r requirements.txt

# 5. Jalankan aplikasi
python app.py
```

Buka browser dan akses: `http://127.0.0.1:5000`

---

## 🔧 Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| **Language** | Python 3.10.14 |
| **ML Framework** | scikit-learn 1.6.1 |
| **Image Processing** | scikit-image, Pillow |
| **Web Framework** | Flask 3.0.3 |
| **WSGI Server** | Gunicorn |
| **Deployment** | Railway |
| **Version Control** | GitHub |
| **Notebook** | Google Colab |

---

## 📓 Alur Pipeline

```
Dataset (750 gambar)
        │
        ▼
Load & Resize (64×64 px)
        │
        ▼
Preprocessing
├── Normalisasi Piksel [0,255] → [0.0,1.0]
├── Cek Imbalance (rasio = 1.00 ✓)
└── Deteksi Outlier via IQR
        │
        ▼
Feature Engineering (80 fitur)
├── Mean & Std RGB (6)
├── Histogram HSV (48)
└── Histogram LBP (26)
        │
        ▼
Feature Scaling (StandardScaler)
        │
        ▼
Split Data (80% Train / 20% Test, stratified)
        │
        ▼
Training Model
├── KNN (K=2, 5-Fold CV)
├── SVM (RBF, C=10)
└── Random Forest (200 trees)
        │
        ▼
Evaluasi (Akurasi, Presisi, Recall, Specificity, F1)
        │
        ▼
McNemar's Test (α=0.05)
        │
        ▼
Deployment Flask → Railway 🚀
```

---

## 📁 Dataset

- **Total**: 750 gambar
- **Distribusi**: 250 gambar per kelas (perfectly balanced)
- **Format**: JPG, JPEG, PNG
- **Ukuran input**: Di-resize ke 64×64 piksel

---

## 🖥️ Tampilan Aplikasi Web

Aplikasi web menggunakan desain **dark mode modern** dengan fitur:
- 🎨 Background glow animasi
- 📤 Drag-and-drop upload gambar
- ⚙️ Pilihan tiga model (KNN, SVM, Random Forest)
- 🔄 Confidence ring animasi
- 📊 Bar chart probabilitas per kelas

---

## 👨‍💻 Author

**Nur Muhammad Anang Febriananto**  
📧 NIM: 230605110103
🏛️ Teknik Informatika — UIN Maulana Malik Ibrahim Malang  
🔗 GitHub: [@brynnstilearning](https://github.com/brynnstilearning)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
  <sub>⭐ Jika project ini bermanfaat, jangan lupa kasih star!</sub>
</div>
