
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
