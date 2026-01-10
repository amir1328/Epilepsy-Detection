import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('epilepsy_federated_dataset.csv')

# 2. Preprocessing
print("Preprocessing...")
# Define target
target_col = 'Seizure_Type_Label'

# Drop other potential targets to prevent leakage
drop_cols = ['Seizure_Type_Label', 'Multi_Class_Label'] 
# Also dropping 'Multi_Class_Label' as it's likely a more granular version of the target or correlated target.

X = df.drop(columns=drop_cols, errors='ignore')
y = df[target_col]

# Split data
print("Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
print("Training Random Forest Classifier...")
# Limiting n_estimators and max_depth initially for speed, can increase later
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) 
rf_model.fit(X_train_scaled, y_train)

# 4. Evaluation
print("Evaluating model...")
y_pred = rf_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Save Model (Optional)
print("Saving model and scaler...")
joblib.dump(rf_model, 'epilepsy_rf_model.pkl')
joblib.dump(scaler, 'epilepsy_scaler.pkl')
print("Done.")
