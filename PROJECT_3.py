# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv(r"C:\Users\kalpa\Downloads\loan_approval_dataset.csv")
df.columns = df.columns.str.strip()  

# Basic Data Exploration
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# EDA 
plt.figure(figsize=(8, 4))
sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Count')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

sns.boxplot(x='education', y='income_annum', data=df)
plt.title("Income Distribution by Education Level")
plt.xticks([0, 1], ['Not Graduate', 'Graduate'])
plt.show()

sns.histplot(df['loan_amount'], kde=True, bins=20)
plt.title('Loan Amount Distribution')
plt.show()

# Feature Engineering
# Log transform to handle skewness
df['loan_amount_log'] = np.log(df['loan_amount'].replace(0, np.nan))
df['total_assets'] = df[['residential_assets_value', 'commercial_assets_value',
                         'luxury_assets_value', 'bank_asset_value']].sum(axis=1)
df['total_assets_log'] = np.log(df['total_assets'].replace(0, np.nan))

# Label Encoding for categorical variables
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])         # Graduate = 1, Not Graduate = 0
df['self_employed'] = le.fit_transform(df['self_employed']) # Yes = 1, No = 0
df['loan_status'] = le.fit_transform(df['loan_status'])     # Approved = 0, Rejected = 1

# Define Features and Target
features = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount_log', 'loan_term', 'cibil_score', 'total_assets_log']
X = df[features]
y = df['loan_status']

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training and Evaluation
# 1. Decision Tree
dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
dtc_acc = accuracy_score(y_test, y_pred_dtc)

# 2. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
nb_acc = accuracy_score(y_test, y_pred_nb)

# 3. Random Forest
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
rfc_acc = accuracy_score(y_test, y_pred_rfc)

# Print Accuracy Scores
print("\nModel Accuracies:")
print(f"Decision Tree Accuracy   : {dtc_acc:.2f}")
print(f"Naive Bayes Accuracy     : {nb_acc:.2f}")
print(f"Random Forest Accuracy   : {rfc_acc:.2f}")

# Classification Report
print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rfc, target_names=['Approved', 'Rejected']))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rfc), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Approved', 'Rejected'],
            yticklabels=['Approved', 'Rejected'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance (Random Forest)
importances = rfc.feature_importances_
features_sorted = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(8, 5))
sns.barplot(x=[x[1] for x in features_sorted], y=[x[0] for x in features_sorted])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Model Comparison Plot
models = ['Decision Tree', 'Naive Bayes', 'Random Forest']
scores = [dtc_acc, nb_acc, rfc_acc]

plt.figure(figsize=(6, 4))
sns.barplot(x=models, y=scores, hue=models, palette='viridis', legend=False)  
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
