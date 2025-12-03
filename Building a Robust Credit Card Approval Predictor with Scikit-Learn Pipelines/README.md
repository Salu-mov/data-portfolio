# 💳 Credit Card Approval Prediction

## Project Overview
This project builds a Machine Learning model to predict credit card approval decisions based on applicant data. Using the **UCI Credit Approval Data Set**, the goal is to automate the decision-making process with high accuracy and interpretability.

## 🛠 Tech Stack
* **Python** (Pandas, NumPy)
* **Scikit-learn** (Pipeline, GridSearchCV, LogisticRegression)
* **Visualization** (Matplotlib, Seaborn)

## 🚀 Key Features
* **End-to-End Pipeline:** Utilized `sklearn.pipeline` to streamline data cleaning (imputation), scaling, encoding, and modeling.
* **Robust Preprocessing:** Handled missing values (`?`) effectively:
    * **Numeric:** Imputed with median & scaled via `StandardScaler`.
    * **Categorical:** Imputed with mode & encoded via `OneHotEncoder`.
* **Hyperparameter Tuning:** Optimized the Logistic Regression model using `GridSearchCV` to find the best `C` and `solver` parameters.
* **Evaluation:** Assessed performance using **Accuracy**, **Confusion Matrix**, and **ROC-AUC Score**.

## 📊 Results
The model achieved strong performance metrics, demonstrating high reliability in distinguishing between approved and denied applications.

| Metric | Score |
| :--- | :--- |
| **Best CV Accuracy** | **87.68%** |
| **Test Accuracy** | **82.00%** |
| **ROC AUC Score** | **0.8965** |

### Model Performance (ROC & Confusion Matrix)
The ROC curve demonstrates the model's ability to maximize true positives while minimizing false positives.
<img width="1400" height="600" alt="resim" src="https://github.com/user-attachments/assets/101fd2fa-0561-4de3-91a8-c8b9026680ab" />


### Feature Importance
The analysis reveals which factors most heavily influence the approval decision (e.g., Prior Default, Credit Score).
<img width="1000" height="600" alt="resim" src="https://github.com/user-attachments/assets/60fc4e2a-c8e6-48ef-ac78-6024fdd7b441" />


## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Salu-mov/Credit-Card-Approval-Prediction.git](https://github.com/Salu-mov/Credit-Card-Approval-Prediction.git)
