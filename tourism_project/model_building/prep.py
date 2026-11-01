import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset and drop index/ID columns
df = pd.read_csv("tourism_project/data/tourism.csv")
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Define target variable for tourism data
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Perform stratified split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Export splits for the model training step
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
