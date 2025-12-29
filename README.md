# 🚚 High-Risk Logistics Detection: Binary Classification with Explainable AI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Machine Learning for Critical Risk Detection in Logistics Operations

## 📊 Key Results

| Metric | Score |
|--------|-------|
| **F1-Score** | 86.23% |
| **Recall** | 99.79% |
| **Precision** | 75.92% |
| **Accuracy** | 75.81% |

**Key Achievement:** Captures 99.8% of all High Risk scenarios.

---

## 🎯 Overview

This repository implements a machine learning framework for detecting high-risk scenarios in logistics operations using:
- **7 ML algorithms** (LightGBM, XGBoost, Gradient Boosting, CatBoost, SVM, Random Forest, Neural Network)
- **SMOTE** for class imbalance handling
- **SHAP** for explainable AI
- **32,065 real-world samples** from Southern California logistics network

---

## 📁 Repository Structure
``````
├── notebooks/          # Python scripts for data processing and modeling
├── models/            # Trained ML models (.pkl files)
├── figures/           # Visualizations and publication figures
│   └── binary/       # Binary classification results
└── data/
    └── results/      # Model performance metrics (CSV files)
``````

---

## 🚀 Quick Start

### Install Dependencies
``````bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost imbalanced-learn shap matplotlib seaborn
``````

### Make Prediction
``````python
import joblib
import numpy as np

# Load best model
model = joblib.load('models/lightgbm_model.pkl')

# Your data (33 features)
X_new = np.array([[...]])  # Replace with your logistics data

# Predict
prediction = model.predict(X_new)
print(f"Risk Level: {'High Risk' if prediction[0] == 0 else 'Non-High Risk'}")
``````

---

## 📊 Dataset

- **32,065 samples** (January 2021 - August 2024)
- **33 features**: temporal, operational, environmental, behavioral
- **Temporal train-test split** (80-20)
- **Class imbalance**: 7.7:1 ratio (High Risk : Low Risk)

### Key Features:
- 📍 Location (GPS, route risk)
- ⏰ Temporal (month, day, hour)
- 🚛 Operational (fuel, loading time, inventory)
- 🌦️ Environmental (weather, traffic, port congestion)
- 👨‍✈️ Behavioral (driver behavior, fatigue)

---

## 🤖 Methodology

### 1. Data Preprocessing
- Temporal split (80-20) to prevent data leakage
- Feature engineering (33 total features)
- StandardScaler normalization
- SMOTE for training set balancing

### 2. Model Selection
Comprehensive evaluation of 7 algorithms:

| Model | F1-Score | Recall | Precision |
|-------|----------|--------|-----------|
| **LightGBM** ⭐ | **0.8623** | **0.9979** | **0.7592** |
| XGBoost | 0.8613 | 0.9955 | 0.7590 |
| Gradient Boosting | 0.8610 | 0.9951 | 0.7588 |

### 3. Explainability (SHAP)
Top 5 Most Important Features:
1. **Month** (0.340) - Seasonal patterns
2. **Day of Week** (0.305) - Operational cycles  
3. **Year** (0.271) - Trend evolution
4. **Hour** (0.087) - Time of day effects
5. **IoT Temperature** (0.054) - Cargo conditions

---

## 📈 Results

### Performance Highlights:
- ✅ **99.79% Recall** - Catches 4,855 out of 4,865 High Risk scenarios
- ✅ **86.23% F1-Score** - Excellent balance
- ✅ **Safety-Critical Focus** - Minimizes missed High Risk cases

### Confusion Matrix:
|  | Predicted Non-High | Predicted High |
|---|-------------------|----------------|
| **Actual Non-High** | 8 | 1,540 |
| **Actual High** | 10 | 4,855 |

---

## 🛠️ Technologies

- Python 3.11
- LightGBM, XGBoost, scikit-learn
- SHAP for explainability
- SMOTE for class balancing
- Matplotlib, Seaborn for visualization

---

## 📝 Citation
``````bibtex
@article{hassan2025logistics,
  title={Machine Learning for Critical Risk Detection in Logistics Operations},
  author={Hassan, Mahbub},
  journal={Transportation Research},
  year={2025}
}
``````

---

## 👥 Author

**Mahbub Hassan**
- 🎓 Transportation Engineering Researcher
- 🏛️ Chulalongkorn University, Bangkok, Thailand
- 📧 mahbub.hassan@ieee.org | 6870376421@student.chula.ac.th
- 🔗 [GitHub](https://github.com/mahbubchula) | [Google Scholar](https://scholar.google.com)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Dataset: Kaggle Logistics and Supply Chain Dataset
- Institution: Chulalongkorn University, Department of Civil Engineering
- Frameworks: LightGBM, SHAP, scikit-learn

---

**⭐ If you find this useful, please star the repository!**
