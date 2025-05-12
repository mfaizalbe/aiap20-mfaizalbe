Machine Learning Pipeline for Predicting Subscription Status
Full Name and Contact Info
Full Name: Muhammad Faizal Bin Ehsan

Email: mdfaizalehsan@gmail.com

Project Overview
This project focuses on predicting whether a customer will subscribe to a service (binary classification). The dataset contains various features about customer behavior, and the goal is to classify whether a customer will subscribe (Yes) or not (No).

The pipeline performs:

Exploratory Data Analysis (EDA) for understanding the data and identifying patterns.

Preprocessing of features, including handling missing data and scaling.

Model Training using multiple classifiers such as Logistic Regression, Decision Tree, Random Forest, XGBoost, and Naive Bayes.

Model Evaluation using various metrics, including Accuracy, Precision, Recall, F1-Score, and ROC AUC.

Folder Structure
yaml

Project Folder Structure:
- .gitHub/                   # GitHub-related configurations (optional)
- src/                        # Folder containing Python scripts/modules for the ML pipeline
  - eda.py               # Main Python script with the pipeline implementation
- run.sh                      # Bash script to execute the pipeline
- requirements.txt            # List of dependencies required to run the pipeline
- README.md                   # This file
- eda.ipynb                   # Jupyter notebook containing exploratory data analysis
Installation and Setup
Prerequisites
Ensure you have Python 3.6+ installed on your system.

Setting up the environment
Clone this repository to your local machine:

bash

git clone https://github.com/your_username/your_repository.git
cd your_repository

Install the required dependencies:

bash

pip install -r requirements.txt
Running the Pipeline
To execute the full pipeline, simply run the following command in your terminal:

bash

./run.sh

This will execute the entire machine learning pipeline, including data loading, preprocessing, model training, and evaluation.

Modifying Parameters
The main hyperparameters for the models (like max_depth, n_estimators, etc.) are defined within the eda.py script. You can modify these parameters to experiment with different configurations.

If you wish to adjust the feature engineering or preprocessing steps, you can modify the code within eda.py where the data is processed.

Logical Flow of the Pipeline
Data Loading and Preprocessing:

The dataset is loaded from a CSV file, missing values are handled, and features are scaled or encoded as needed.

Exploratory Data Analysis (EDA):

Descriptive statistics, correlation analysis, and visualizations are generated in the eda.ipynb notebook to understand the data better.

Model Training and Evaluation:

Multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost, Naive Bayes) are trained and evaluated using cross-validation.

The evaluation metrics include accuracy, precision, recall, F1-score, and ROC AUC.

Model Tuning:

Hyperparameter tuning and feature engineering are applied to optimize model performance.

Key Findings from EDA
Imbalanced Dataset
The dataset contains imbalanced classes, with more No (non-subscribers) than Yes (subscribers). This created challenges for model performance, especially in correctly predicting the minority class (Yes).

Feature Correlation
Key features such as Age, Campaign Calls, and Previous Contact Days showed strong correlations with the target variable (Subscription Status), and were thus selected as important features in the models.

Class Imbalance Handling
SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the dataset and improve the recall for the minority class (Yes).

Feature Engineering and Processing
Data Preprocessing Steps:
Categorical Features: Categorical features like Occupation, Marital Status, and Education Level were encoded using one-hot encoding.

Continuous Features: Features like Campaign Calls were binned into discrete categories for better interpretability.

Class Balancing: SMOTE was applied to generate synthetic samples for the minority class (Yes).

Model Selection and Evaluation
Model Selection
We trained and evaluated the following models:

Logistic Regression: Chosen for its simplicity and interpretability.

Decision Tree: A non-linear model that can handle complex relationships between features.

Random Forest: An ensemble model that mitigates overfitting and improves robustness.

XGBoost: An advanced gradient boosting model known for its high performance and ability to handle imbalanced datasets.

Naive Bayes: A baseline model to compare against other more complex models.

Evaluation Metrics
The models were evaluated using the following metrics:

Accuracy: The proportion of correct predictions.

Precision: The proportion of true positive predictions out of all positive predictions.

Recall: The proportion of true positives out of all actual positives.

F1-Score: The harmonic mean of precision and recall.

ROC AUC: Area under the Receiver Operating Characteristic curve, a metric that captures the model's ability to distinguish between the classes.

Model Evaluation Summary
Logistic Regression:

Accuracy: 89.55%

ROC AUC: 0.72

Strong accuracy but struggles with recall for the minority class (Yes).

XGBoost:

Accuracy: 89.31%

ROC AUC: 0.73

Slight improvement in AUC over Logistic Regression but still struggles with recall for Yes class.

Random Forest:

Accuracy: 87.39%

ROC AUC: 0.67

Performs slightly better than Decision Tree but still biased towards the majority class.

Decision Tree:

Accuracy: 84.46%

ROC AUC: 0.60

Struggles the most with the imbalanced dataset and has low recall for the minority class.

Naive Bayes:

Accuracy: 89.43%

Owner avatar



Offers good performance, with slightly better recall for Yes compared to Logistic Regression.

Conclusion and Future Work
Best Performing Model: XGBoost and Naive Bayes showed strong performance with ROC AUCs of 0.73, though there is still room for improvement in recall for the minority class (Yes).

Next Steps:

Hyperparameter tuning: Further tuning of the models, especially XGBoost, could improve recall for the minority class.

Additional Data: More data, especially for the minority class (Yes), would help improve model performance.

Model Deployment: The models can be deployed as REST APIs for real-time predictions.

Deployment Considerations
Model Deployment: The trained models can be deployed as REST APIs using frameworks like Flask or FastAPI.

Continuous Learning: As new data becomes available, models can be retrained to adapt to changing customer behaviors.

Acknowledgements
Dataset: https://techassessment.blob.core.windows.net/aiap20-assessment-data/bmarket.db

Libraries: Pandas, Scikit-learn, XGBoost, etc.
