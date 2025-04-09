import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 🔹 Load dataset
df = pd.read_csv('lung_cancer_data.csv')

# 🔹 Encode categorical columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# 🔹 Features and target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# 🔹 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Define ML models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

# 🔹 Train & Evaluate
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label=1),
        'Recall': recall_score(y_test, y_pred, pos_label=1),
        'F1 Score': f1_score(y_test, y_pred, pos_label=1)
    })

# 🔹 Create DataFrame for metrics
metrics_df = pd.DataFrame(results)
print("\n🔍 Comparing ML Models:\n")
print(metrics_df)

# 🔹 Save best model
best_model_name = metrics_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]

# 🔹 Create folder to save model if not exist
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/lung_cancer_model.pkl')
print(f"\n✅ Best model '{best_model_name}' saved to models/lung_cancer_model.pkl")

# 🔹 Plotting metrics
metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()

# 🔹 Save plot
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/model_comparison.png')
plt.show()
