import joblib
import pandas as pd

model = joblib.load("../model/loan_model.pkl")

sample = pd.DataFrame({
    "ApplicantIncome":[5000],
    "CoapplicantIncome":[2000],
    "LoanAmount":[150],
    "Loan_Amount_Term":[360],
    "Credit_History":[1],
    "Education":[1],
    "Property_Area":[2]
})

prediction = model.predict(sample)

print("Loan Approval Prediction:", prediction[0])
