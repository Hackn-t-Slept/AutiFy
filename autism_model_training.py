from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('train.csv')

# Label encoding
df['gender'] = df['gender'].map({'m': 0, 'f': 1})
df['jaundice'] = df['jaundice'].map({'no': 0, 'yes': 1})
df['austim'] = df['austim'].map({'no': 0, 'yes': 1})
df['used_app_before'] = df['used_app_before'].map({'no': 0, 'yes': 1})
df['Class/ASD'] = df['Class/ASD'].map({'NO': 0, 'YES': 1})

# Features and target
features = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'jaundice', 'austim', 'used_app_before'
]
X = df[features]
y = df['Class/ASD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with ONLY sklearn
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model using joblib
joblib.dump(model, 'autism_prediction_model.pkl')

print("✅ Model saved as autism_prediction_model.pkl")
