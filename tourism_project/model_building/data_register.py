
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    'Unnamed: 0', 'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 
    'CityTier', 'DurationOfPitch', 'Occupation', 'Gender', 
    'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 
    'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 
    'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
