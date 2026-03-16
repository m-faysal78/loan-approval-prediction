import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess

df = load_data("../data/loan_data.csv")

X_train, X_test, y_train, y_test = preprocess(df)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

os.makedirs("../model", exist_ok=True)

joblib.dump(model, "../model/loan_model.pkl")

print("Loan approval model trained successfully")
