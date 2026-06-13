# ⚖️ Model Fairness, Bias & Explainability Analysis

A machine learning fairness and explainability project analyzing a **Random Forest customer churn classifier** using SHAP, LIME, and bias mitigation techniques — part of the **InternSpark AI Internship Task 3**.

---

## 📌 Project Overview

This project investigates:
- **Explainability** — understanding *why* the model makes predictions (global & local)
- **Bias Detection** — identifying unfair treatment across sensitive groups (gender, age)
- **Bias Mitigation** — applying SMOTE to reduce disparity between groups

**Dataset:** Telco Customer Churn (`customer_churn_data.csv`)  
**Model:** RandomForestClassifier  
**Sensitive Attributes:** `gender` (Male/Female), `SeniorCitizen` (0/1)

---

## 🗂️ Project Structure

```
model-fairness-explainability/
│
├── notebook/
│   └── task3.ipynb                  # Full analysis notebook
│
├── data/
│   └── customer_churn_data.csv      # Dataset
│
├── reports/
│   └── fairness_analysis.md         # Detailed fairness report
│
├── screenshots/
│   ├── screenshot 1.png             # SHAP / model output
│   └── Screenshot 2.png             # Bias comparison plots
│
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🔍 Analysis Breakdown

### 1. 🌳 Feature Importance
Top features driving churn predictions: `tenure`, `MonthlyCharges`, `Contract`, and more — visualized as a horizontal bar chart.

### 2. 🌐 Global Explainability — SHAP
- **Summary Plot**: Shows impact of each feature on churn predictions
- **Bar Plot**: Mean absolute SHAP values per feature
- **Force Plot**: Local explanation for a single prediction

### 3. 🔬 Local Explainability — LIME
- Explains individual predictions using top 10 contributing features

### 4. ⚠️ Bias Detection

**Before Mitigation:**

| Group | Accuracy | FPR | TPR | Pred Rate |
|---|---|---|---|---|
| Female (gender_Male=0) | ~0.80 | 0.15 | 0.45 | 0.25 |
| Male (gender_Male=1) | ~0.78 | 0.18 | 0.42 | 0.27 |
| Non-Senior (SeniorCitizen=0) | ~0.82 | 0.14 | 0.44 | 0.24 |
| Senior (SeniorCitizen=1) | ~0.75 | 0.20 | 0.48 | 0.30 |

> ⚠️ Males and Seniors show higher False Positive Rates — indicating bias.

### 5. 🛠️ Bias Mitigation — SMOTE

**After Mitigation:**

| Group | Accuracy | FPR | TPR | Pred Rate |
|---|---|---|---|---|
| Female | ~0.79 | 0.16 | 0.46 | 0.26 |
| Male | ~0.80 | 0.16 | 0.45 | 0.26 |
| Non-Senior | ~0.81 | 0.15 | 0.45 | 0.25 |
| Senior | ~0.78 | 0.17 | 0.46 | 0.27 |

✅ FPR disparity reduced from **0.04 → 0.01** for gender — significant improvement!

---

## 📈 Screenshots

| SHAP Analysis | Bias Comparison |
|---|---|
| ![Screenshot 1](screenshots/screenshot%201.png) | ![Screenshot 2](screenshots/Screenshot%202.png) |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/model-fairness-explainability.git
cd model-fairness-explainability
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook
```
Open `notebook/task3.ipynb` and run all cells sequentially.

---

## 📋 Notebook Walkthrough

| Cells | Description |
|---|---|
| 1–2 | Data loading, preprocessing, model training |
| 3 | Feature importance plot |
| 4 | Install additional libraries if needed |
| 5–7 | SHAP global summary, bar plot, force plot |
| 8 | LIME local explanation |
| 9–11 | Bias checks (gender, SeniorCitizen) |
| 12–15 | SMOTE mitigation + post-mitigation metrics |
| Final | Full comparison: bias before vs. after |

---

## 🛠️ Technologies Used

- Python 3.x
- scikit-learn (RandomForestClassifier)
- SHAP — global & local explainability
- LIME — local prediction explanations
- imbalanced-learn (SMOTE) — bias mitigation
- pandas, numpy, matplotlib, seaborn
- Jupyter Notebook

---

## 📄 Full Report

See [`reports/fairness_analysis.md`](reports/fairness_analysis.md) for the complete detailed analysis.

---

## 👤 Author

Internship Project — InternSpark | Artificial Intelligence Domain | Task 3
