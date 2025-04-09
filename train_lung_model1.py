import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 📂 Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# 📄 Load and preprocess data
df = pd.read_csv('lung_cancer_data.csv')

# Target conversion
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Encode features
X = df.drop('LUNG_CANCER', axis=1)
X_encoded = pd.get_dummies(X)
y = df['LUNG_CANCER']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 🧠 Define ML models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Naive Bayes': GaussianNB()
}

# 📊 Compare model metrics
metrics = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0)
    })

# 📄 Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)
print("\n🔍 Model Performance Comparison:")
print(metrics_df)

# 📈 Save performance plot
metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', figsize=(12, 6))
plt.title("Model Comparison Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/model_comparison_metrics.png')
plt.show()

# 💾 Save best model
best_model_name = metrics_df.sort_values(by='F1 Score', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'models/lung_cancer_model.pkl')
print(f"\n✅ Best model saved: {best_model_name}")

# 📊 Feature Importance (Random Forest)
rf = models['Random Forest']
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', dodge=False, legend=False)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig('plots/feature_importance_rf.png')
plt.show()

# 📌 SHAP Interpretability
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# ✅ Dynamically handle shap_values shape
if isinstance(shap_values, list):
    shap_vals_to_plot = shap_values[1]  # For classification, class 1
else:
    shap_vals_to_plot = shap_values     # For regression or auto-detected cases

# SHAP Summary Plot - Bar
shap.summary_plot(shap_vals_to_plot, X_test, plot_type="bar", show=False)
plt.title('SHAP Summary (Bar) - Random Forest')
plt.tight_layout()
plt.savefig('plots/shap_summary_bar.png')
plt.show()

# SHAP Beeswarm
shap.summary_plot(shap_vals_to_plot, X_test, show=False)
plt.title('SHAP Summary (Beeswarm) - Random Forest')
plt.tight_layout()
plt.savefig('plots/shap_beeswarm.png')
plt.show()
