# Loan_Approval_Prediction_ML
Machine learning project for predicting loan approval status using Decision Tree, Naive Bayes, and Random Forest models with feature engineering, scaling, and model comparison.

# 💳 Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be **approved or rejected** based on applicant details and financial information.  
It applies **Decision Tree**, **Naive Bayes**, and **Random Forest** classifiers to perform prediction and compares their performance.

---

## 📌 Features of the Project
- **Data Preprocessing**
  - Removes whitespace from column names.
  - Handles missing values.
  - Encodes categorical variables with `LabelEncoder`.
  - Applies log transformation for skewed features.
  - Creates new features like **total assets**.
  - Scales features using `StandardScaler`.

- **Exploratory Data Analysis (EDA)**
  - Loan status distribution plot.
  - Income distribution by education level.
  - Loan amount distribution.

- **Model Training**
  - Decision Tree Classifier
  - Naive Bayes Classifier
  - Random Forest Classifier

- **Model Evaluation**
  - Accuracy scores for all models.
  - Classification report for Random Forest.
  - Confusion matrix heatmap.
  - Feature importance plot for Random Forest.
  - Model accuracy comparison plot.
