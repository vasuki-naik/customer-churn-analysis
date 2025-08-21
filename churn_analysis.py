# Customer Churn Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv(r"D:\.vscode\D_new\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Encode Target Variable (Churn: Yes/No -> 1/0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate target and features
target = 'Churn'
X = df.drop(columns=target)
y = df[target]

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Preprocessing (fit on training data only)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 5. Handle Imbalance using SMOTE (only on training set)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_balanced.value_counts())

# Function to train & evaluate model
def train_evaluate_model(model, model_name):
    print(f"\n--- Training {model_name} ---")
    
    # Train model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predict
    y_pred = model.predict(X_test_processed)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_processed)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"{model_name} ROC AUC: {roc_auc:.4f}")
    
    # Feature Importance (if available)
    if model_name in ['Random Forest', 'XGBoost']:
        # Get feature names
        feature_names = []
        if num_cols:
            feature_names.extend(preprocessor.named_transformers_['num'].get_feature_names_out(num_cols))
        if cat_cols:
            feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
        
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.tight_layout()
        plt.show()
    
    return model

# 6. Models
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# 7. Train & Evaluate
logreg_model = train_evaluate_model(log_reg, "Logistic Regression")
rf_model = train_evaluate_model(rf, "Random Forest")
xgb_model = train_evaluate_model(xgb_clf, "XGBoost")

# 8. Save Best Model (assuming XGBoost performs best)
joblib.dump(rf_model, 'rf_churn_model.joblib')

print("\nModel saved as 'xgb_churn_model.joblib'")
