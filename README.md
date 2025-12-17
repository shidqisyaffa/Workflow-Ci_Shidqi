# Workflow-Ci_Shidqi

## 📌 Deskripsi

Repository ini berisi **workflow CI** menggunakan **MLflow Project** untuk melakukan **re-training model machine learning** secara otomatis ketika trigger CI dipantik.

**Level Implementasi**: SKILLED (3 Poin)

## 🛠️ Tech Stack

- **Python 3.12.7**
- **MLflow 2.19.0**
- **GitHub Actions**
- **scikit-learn** (RandomForestClassifier)

## 📂 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
├── MLProject/
│   ├── MLProject               # MLflow project definition
│   ├── conda.yaml              # Conda environment
│   ├── modelling.py            # Training script
│   └── gpu_data_processed.csv  # Dataset
└── README.md
```

## ⚙️ CI Workflow Steps (SKILLED Level)

1. **Set up job** - GitHub Actions runner
2. **Checkout repository** - `actions/checkout@v3`
3. **Set up Python 3.12.7** - `actions/setup-python@v4`
4. **Check environment** - Validasi Python & directory
5. **Install dependencies** - MLflow & scikit-learn
6. **Set MLflow Tracking URI** - Konfigurasi tracking lokal
7. **Run MLflow Project** - Training model
8. **Get latest MLflow run_id** - Extract run info
9. **Verify dependencies** - Validasi packages
10. **Upload artifact** - Upload ke GitHub Artifacts

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Install dependencies
pip install mlflow==2.19.0 pandas numpy scikit-learn

# 2. Jalankan MLflow Project
cd MLProject
mlflow run . --env-manager=local

# 3. Lihat hasil di folder mlruns/
```

## 🔄 Trigger CI

Workflow akan otomatis berjalan saat:
- **Push** ke branch `main` atau `master`
- **Pull Request** ke branch `main` atau `master`
- **Manual trigger** via `workflow_dispatch`

## 📦 Artifacts

Model artifacts akan di-upload ke GitHub Artifacts dan tersedia selama 30 hari.

## 👤 Author

- **Nama**: Shidqi Syaffa
- **Email**: musyaffashidqi@gmail.com
- **GitHub**: [@shidqisyaffa](https://github.com/shidqisyaffa)
