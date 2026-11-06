# --------------------------------------------------
# STEP 1: IMPORT ALL NECESSARY PACKAGES
# --------------------------------------------------

import numpy as np
import pandas as pd

# Import classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import XGBoost classifier
from xgboost import XGBClassifier

# Import function to split dataset
from sklearn.model_selection import train_test_split

# Import preprocessing tools
from sklearn.preprocessing import StandardScaler

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Import SMOTE to handle class imbalance
from imblearn.over_sampling import SMOTE

# Import Pipeline
from imblearn.pipeline import Pipeline

# Import Path to locate files
from pathlib import Path

# Import Joblib to save trained models
import joblib
# --------------------------------------------------
# STEP 2: READ THE CSV DATASET
# --------------------------------------------------

# Define the path to the credit card dataset
data = Path("dataset/creditcard.csv")

# Read the CSV file using Pandas
df = pd.read_csv(data)

# Display the first five records
print(df.head())
# --------------------------------------------------
# STEP 3: PERFORM EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------

# Display the number of rows and columns
print("Dataset shape:")
print(df.shape)

# Display dataset information
print("\nDataset information:")
df.info()

# Display statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Count duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
# --------------------------------------------------
# STEP 4: SEPARATE FEATURES AND TARGET
# --------------------------------------------------

# Features (X) contain input columns
# Remove the target column "Class"
X = df.drop("Class", axis=1)

# Target (y) contains the output column
y = df["Class"]
# --------------------------------------------------
# STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

# Split the dataset into 80% training and 20% testing data

# random_state=42 ensures reproducible results
# stratify=y preserves the fraud and legitimate transaction ratio

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the shape of training and testing data
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# --------------------------------------------------
# STEP 6: TRAIN THE LOGISTIC REGRESSION MODEL
# --------------------------------------------------

# Create a pipeline with scaling, SMOTE and Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model using training data
model.fit(X_train, y_train)

print("\nLogistic Regression model training completed successfully!")
# --------------------------------------------------
# STEP 7: TEST AND EVALUATE THE MODEL
# --------------------------------------------------

# Predict transaction classes using testing data
y_test_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)

# Calculate precision
precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

# Calculate recall
recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

# Calculate F1 score
f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

# Calculate ROC-AUC score
y_test_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_test_prob)

# Display evaluation results
print("\nModel Evaluation Results")
print("------------------------")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Save confusion matrix for Streamlit visualization
cm = confusion_matrix(y_test, y_test_pred)
joblib.dump(cm, "confusion_matrix.joblib")
# --------------------------------------------------
# STEP 8: SAVE THE TRAINED MODEL AND FEATURE NAMES
# --------------------------------------------------

# Save the trained Logistic Regression model
joblib.dump(model, "model.joblib")

# Save the feature names in the same order used during training
joblib.dump(list(X.columns), "feature_names.joblib")

print("\nModel saved as model.joblib")
print("Feature names saved as feature_names.joblib")
# --------------------------------------------------
# STEP 9: COMPARE CLASSIFICATION ALGORITHMS
# --------------------------------------------------

# Create classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}

# Store model comparison results
comparison_results = []

# Train and evaluate each algorithm
for name, classifier in models.items():

    comparison_model = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier)
    ])

    comparison_model.fit(X_train, y_train)

    predictions = comparison_model.predict(X_test)

    comparison_results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

# Display comparison table
comparison_df = pd.DataFrame(comparison_results)

print("\nCLASSIFICATION ALGORITHM COMPARISON")
print("-----------------------------------")
print(comparison_df.to_string(index=False))

# Save comparison results
comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)