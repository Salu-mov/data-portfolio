# 💳 Credit Card Approval Prediction

## Project Overview
This project builds a Machine Learning model to predict credit card approval decisions based on applicant data. Using the **UCI Credit Approval Data Set**, the goal is to automate the decision-making process with high accuracy and interpretability.

## 🛠 Tech Stack
* **Python** (Pandas, NumPy)
* **Scikit-learn** (Pipeline, GridSearchCV, LogisticRegression)
* **Visualization** (Matplotlib, Seaborn)

## 🚀 Key Features
* **End-to-End Pipeline:** Utilized `sklearn.pipeline` to streamline data cleaning (imputation), scaling, encoding, and modeling.
* **Robust Preprocessing:** Handled missing values (`?`) and applied `StandardScaler` for numeric data and `OneHotEncoder` for categorical data.
* **Hyperparameter Tuning:** Optimized the Logistic Regression model using `GridSearchCV` to find the best `C` and `solver` parameters.
* **Evaluation:** Assessed performance using **Accuracy**, **Confusion Matrix**, and **ROC-AUC Score**.

## 📊 Results
* **Best CV Accuracy:** [Insert your score, e.g., 0.8500]
* **ROC AUC Score:** [Insert your score, e.g., 0.9100]
* The model successfully identifies key factors influencing approval (Feature Importance analysis included).
