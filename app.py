import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("titanic.csv")

# Select EXACT 6 features
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]

# Encode sex
X['Sex'] = X['Sex'].map({'male': 1, 'female': 0})

# Fill missing values
X.fillna(X.mean(), inplace=True)

y = df['Survived']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)
