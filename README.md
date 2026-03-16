# Loan Approval Prediction System

Machine learning system that predicts loan approval outcomes based on borrower attributes such as income, education, credit history, and loan characteristics.

## Problem Statement

Financial institutions evaluate loan applications by analyzing multiple borrower attributes. This project builds a classification model that predicts whether a loan should be approved using historical loan data.

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Seaborn  
Matplotlib  

## Project Structure

loan-approval-prediction

data/
loan_data.csv

src/
preprocess.py
train.py
predict.py

## Workflow

1. Data cleaning and preprocessing
2. Handling missing values
3. Feature encoding using LabelEncoder
4. Model training using Random Forest
5. Prediction for new loan applications

## Model Used

Random Forest Classifier

Reason for selection:
- Handles nonlinear relationships
- Works well with tabular datasets
- Robust to feature scaling

## How to Run

Install dependencies

pip install -r requirements.txt

Train the model

python src/train.py

Predict loan approval

python src/predict.py

## Example Prediction

Loan Approval Prediction: Approved

## Future Improvements

Add model comparison (Logistic Regression, XGBoost)

Implement feature importance visualization

Build a web interface using FastAPI or Streamlit

Deploy using Docker
