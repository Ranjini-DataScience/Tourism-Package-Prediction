
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the raw dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Removing unnecessary columns
columns_to_drop = ['Unnamed: 0', 'CustomerID']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

# Clean data values: fixing 'fe male' typo in Gender as 'female' and merge MaritalStatus categories 'Single' and ;Unmarried' to 'Single'
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'fe male': 'Female'})

if 'MaritalStatus' in df.columns:
    df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# Defining independent variables (X) and dependent variable (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Splitting the data into training and testing sets (with stratify to keep class ratio consistent)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Saving split datasets  
Xtrain.to_csv("tourism_project/model_building/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/model_building/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/model_building/ytrain.csv", index=False)
ytest.to_csv("tourism_project/model_building/ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("TypeofContact values kept as:", sorted(X["TypeofContact"].dropna().unique()))
