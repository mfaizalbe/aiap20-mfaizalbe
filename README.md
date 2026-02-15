
# 💡 Machine Learning Pipeline for Predicting Subscription Status

**Author:** Muhammad Faizal Bin Ehsan
📧 **Email:** [mdfaizalehsan@gmail.com](mailto:mdfaizalehsan@gmail.com)

---

## 📌 Project Overview

This project aims to predict whether a customer will subscribe to a service (binary classification). The dataset includes various features related to customer behaviour. The goal is to classify customers as either **subscribed ("Yes")** or **not subscribed ("No")**.

### The pipeline covers:

* **Exploratory Data Analysis (EDA):** Understanding patterns and insights.
* **Data Preprocessing:** Handling missing values, encoding, and scaling.
* **Model Training:** Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes.
* **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC AUC.

---

## 📁 Project Structure

```
aiap20-mfaizalbe/
├── .github/             # GitHub-specific configurations (optional)
├── src/                 # Python scripts/modules for the ML pipeline
│   └── eda.py           # Core implementation of the pipeline
├── eda.ipynb            # Jupyter notebook with EDA visuals and insights
├── run.sh               # Bash script to execute the pipeline
├── requirements.txt     # List of Python dependencies
└── README.md            # Project documentation (this file)
```

---

## ⚙️ Installation and Setup

### ✅ Prerequisites

* Python 3.6 or higher

### 📥 Clone the repository

```bash
git clone https://github.com/mfaizalbe/aiap20-mfaizalbe.git
cd aiap20-mfaizalbe
```

### 📦 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

To run the complete pipeline:

```bash
./run.sh
```

This script runs the entire process including:

* Data loading
* Preprocessing
* Model training
* Evaluation

---

## 🛠 Modifying Parameters

* Modify hyperparameters (e.g., `max_depth`, `n_estimators`) inside `src/eda.py`.
* Adjust feature engineering or preprocessing directly in `eda.py`.

---

## 🔄 Pipeline Workflow

### 📂 Data Loading & Preprocessing

* Load dataset from CSV
* Handle missing values
* Encode categorical features and scale numerical ones

### 📊 Exploratory Data Analysis (EDA)

* Descriptive statistics
* Correlation matrix
* Visualizations (in `eda.ipynb`)

### 🤖 Model Training & Evaluation

* Train: Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes
* Evaluate using:

  * **Accuracy**
  * **Precision**
  * **Recall**
  * **F1-Score**
  * **ROC AUC**

### 🔧 Model Tuning

* Basic hyperparameter tuning performed
* SMOTE was tested for class balancing but not used in the final pipeline due to poorer performance

---

## 📈 Key EDA Findings

* **Imbalanced Dataset:** The dataset has more "No" (non-subscribers) than "Yes" (subscribers), which challenges recall for the minority class.
* **Important Features:** Age, number of campaign calls, and previous contact duration were significantly correlated with subscription outcomes.
* **Class Imbalance Note:** SMOTE was explored but discarded, as it negatively affected model performance.

---

## 🔍 Feature Engineering Summary

* **Categorical Encoding:** One-hot encoding was applied to features such as Occupation, Marital Status, and Education.
* **Discretization:** Continuous features (e.g., campaign calls) were binned into categories to enhance interpretability.
* **Imbalance Handling:** The final pipeline did not use oversampling techniques due to performance degradation.

---

## 🧪 Model Summary

| Model               | Accuracy | ROC AUC | Notes                                                        |
| ------------------- | -------- | ------- | ------------------------------------------------------------ |
| Logistic Regression | 89.55%   | 0.72    | High accuracy, but limited recall for minority class         |
| XGBoost             | 89.31%   | 0.73    | Strong AUC, best overall trade-off                           |
| Random Forest       | 87.39%   | 0.67    | Robust, but biased toward majority class                     |
| Decision Tree       | 84.46%   | 0.60    | Most affected by imbalance                                   |
| Naive Bayes         | 89.43%   | —       | Competitive accuracy, slightly better recall for "Yes" class |

---

## ✅ Conclusion and Next Steps

* **Best Models:** XGBoost and Naive Bayes showed the strongest ROC AUC performance.
* **Areas for Improvement:**

  * Tune hyperparameters further (especially for XGBoost)
  * Collect additional data for minority class ("Yes")
  * Consider alternative balancing strategies (e.g., class weights)

---

## 🚀 Future Deployment

* **Model Deployment:** Models can be deployed using Flask or FastAPI as REST APIs.
* **Continuous Learning:** Future data can be integrated to retrain and improve performance over time.

---

## 🙏 Acknowledgements

* **Dataset:** [Bank Marketing Dataset](https://techassessment.blob.core.windows.net/aiap20-assessment-data/bmarket.db)
* **Libraries Used:** `Pandas`, `Scikit-learn`, `XGBoost`, `Matplotlib`, `Seaborn`, `NumPy`

