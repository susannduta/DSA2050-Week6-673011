# DSA2050 Week 6: Regional Sales Performance & Predictive Demand Forecasting

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline developed for **DSA2050 (Week 6\)** to predict retail sales revenue (`Sales_KES`) across multiple regional distribution centers, customer demographics, and product categories in Kenya.

This project implements leakage-free preprocessing pipelines, baseline comparison models, regularized linear estimators, non-parametric decision trees, and 5-fold cross-validation routines.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)  
2. [Data Architecture & Dictionary](#-data-architecture--dictionary)  
3. [Pipeline Architecture](#-pipeline-architecture)  
4. [Model Benchmark & Performance](#-model-benchmark--performance)  
5. [Key Analytical Insights](#-key-analytical-insights)  
6. [Repository Structure](#-repository-structure)  
7. [Installation & Setup](#-installation--setup)  
8. [Reproducibility & Quick Start](#-reproducibility--quick-start)  
9. [License & Citation](#-license--citation)

---

## 🏢 Project Overview

Predicting regional sales in Kenyan Shillings (KES) allows supply chain managers and financial analysts to optimize inventory levels, direct advertising budgets efficiently, and set realistic revenue targets.

This repository provides an automated machine learning workflow designed to:

- Evaluate baseline strategies against complex supervised learning algorithms.  
- Enforce strict statistical validation via **5-Fold Cross-Validation** and independent **80/20 train-test splits**.  
- Prevent data leakage through unified scikit-learn `Pipeline` and `ColumnTransformer` constructs.

---

## 📊 Data Architecture & Dictionary

The underlying dataset (`DSA2050_Week6_Sales_Data.csv`) contains $N=320$ verified retail records with 6 explanatory features and 1 continuous target variable.

| Attribute | Type | Role | Description | Range / Categories |
| :---- | :---: | :---: | :---- | :---- |
| `AdvertisingSpend_KES` | Continuous | Feature | Regional marketing investment (KES) | $5,478-69,746$ |
| `StoreVisits` | Continuous | Feature | Total foot traffic recorded at retail location | $84-848$ |
| `DiscountPercent` | Continuous | Feature | Average promotional discount applied | $0\% - 25\%$ |
| `Region` | Nominal | Feature | Operational geographical tier | `Nairobi`, `Coast`, `Rift Valley`, `Western` |
| `CustomerSegment` | Nominal | Feature | Purchasing group profile | `Retail`, `Wholesale`, `Corporate` |
| `ProductCategory` | Nominal | Feature | Inventory category | `Electronics`, `Apparel`, `Home`, `Groceries` |
| **`Sales_KES`** | **Continuous** | **Target ($y$)** | **Gross sales revenue achieved (KES)** | **$17,805-295,825$** |

---

## ⚙️ Pipeline Architecture

To prevent data leakage during preprocessing and hyperparameter evaluation, all transformations are encapsulated inside a scikit-learn `ColumnTransformer`.

```
                                  ┌────────────────────────┐
                                  │   Raw Feature Inputs   │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
                      ▼                                               ▼
          [ Numerical Columns ]                           [ Categorical Columns ]
   (Spend, Visits, Discount)                         (Region, Segment, Category)
                      │                                               │
                      ▼                                               ▼
         SimpleImputer(median)                           SimpleImputer(most_frequent)
                      │                                               │
                      ▼                                               ▼
              StandardScaler()                             OneHotEncoder(ignore)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                                   [ Combined Feature Vector ]
                                              │
                                              ▼
                                   [ Machine Learning Model ]
                              (Ridge / OLS / Decision Tree / Baseline)
```

1. **Numerical Pipeline:** Missing values imputed via column median; values standardized ($z=\frac{x-\mu }{\sigma }$) to balance gradient scaling.  
2. **Categorical Pipeline:** Missing values imputed using the most frequent mode; variables transformed via `OneHotEncoder(handle_unknown='ignore')` to prevent dimensionality crashes on unobserved test levels.

---

## 📈 Model Benchmark & Performance

Models were trained on 80% of the dataset ($N_{\text{train}} = 256$) and evaluated on an unobserved holdout test set ($N_{\text{test}} = 64$) with a fixed seed (`RANDOM_STATE = 42`).

### Holdout Test Metrics

| Rank | Estimator | MAE (KES) ↓ | RMSE (KES) ↓ | ${R}^{2}$ Score ↑ | Notes |
| :---: | :---- | :---: | :---: | :---: | :---- |
| 🥇 | **Ridge Regression ($\alpha =1.0$)** | **13,714.40** | **17,458.82** | **0.8705** | **Best overall generalization** |
| 🥈 | Ordinary Least Squares (OLS) | 13,757.84 | 17,469.69 | 0.8703 | Unregularized baseline |
| 🥉 | Decision Tree (`depth=4`) | 23,775.92 | 28,767.64 | 0.6484 | High variance / Step-function split |
| 4 | Median Baseline | 39,146.75 | 48,649.47 | \-0.0056 | Central tendency naive model |
| 5 | Mean Baseline | 39,197.16 | 48,716.43 | \-0.0084 | Central tendency naive model |

### 5-Fold Cross-Validation Metrics (Training Set)

| Estimator | Mean CV MAE (KES) | Mean CV RMSE (KES) | Mean CV ${R}^{2}$ |
| :---- | :---: | :---: | :---: |
| **Ridge Regression ($\alpha =1.0$)** | **15,971.37** | **20,103.54** | **0.8479** |
| Linear Regression (OLS) | 15,976.77 | 20,126.77 | 0.8475 |
| Decision Tree (`depth=4`) | 24,498.47 | 30,608.68 | 0.6506 |

---

## 💡 Key Analytical Insights

- **Linear Primacy:** Linear configurations significantly outperform tree-based methods due to strong continuous linear relationships between advertising expenditure, foot traffic, and gross revenue.  
- **L2 Regularization Advantage:** Ridge Regression ($\alpha =1.0$) effectively minimizes variance caused by multi-collinearity among high-traffic store locations, providing superior stability across cross-validation folds.  
- **Failure of Baseline Models:** Naïve central-tendency estimators yield negative ${R}^{2}$ values, confirming that variance in sales cannot be modeled without accounting for regional and foot-traffic covariates.

---

## 📁 Repository Structure

```
.
├── DSA2050_Week6_Sales_Data.csv           # Primary dataset (320 records, 7 attributes)
├── DSA2050_Week6_Notebook.ipynb           # Executable Jupyter Notebook with full analysis
├── DSA2050_Week6_Sales_Performance_Report.pdf # Comprehensive pdf analysis report
└── README.md                              # Technical project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.13+**  
- `pip` or `conda` package manager

### Environment Setup

```sh
# Clone the repository
git clone https://github.com/susannduta/DSA2050-Week6-673011.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib jupyter
```

---

## 🚀 Reproducibility & Quick Start

Execute the complete training pipeline directly in Python:

```py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load Data
df = pd.read_csv("DSA2050_Week6_Sales_Data.csv")
X = df.drop(columns="Sales_KES")
y = df["Sales_KES"]

# 2. Build Preprocessor
num_cols = ["AdvertisingSpend_KES", "StoreVisits", "DiscountPercent"]
cat_cols = ["Region", "CustomerSegment", "ProductCategory"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

# 3. Construct & Train Pipeline
pipeline = Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=1.0))])

pipeline.fit(X, y)
print(
    f"Model trained successfully. Model R^2 score on full data: {pipeline.score(X, y):.4f}"
)
```

---

&nbsp;