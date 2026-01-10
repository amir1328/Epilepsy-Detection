import pandas as pd

# Load dataset
file_path = 'epilepsy_federated_dataset.csv'
print(f"Loading {file_path}...")
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Basic Structure
print("\n--- Dataset Structure ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n--- Column Names ---")
print(df.columns.tolist())

# Missing Values
print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Label Distribution
print("\n--- Target Variable Distribution ---")
if 'Seizure_Type_Label' in df.columns:
    print("\nSeizure_Type_Label counts:")
    print(df['Seizure_Type_Label'].value_counts(normalize=True))
    print(df['Seizure_Type_Label'].value_counts())

if 'Multi_Class_Label' in df.columns:
    print("\nMulti_Class_Label counts:")
    print(df['Multi_Class_Label'].value_counts(normalize=True))
    print(df['Multi_Class_Label'].value_counts())

print("\n--- Done ---")
