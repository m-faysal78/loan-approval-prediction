import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):

    df = df.dropna()

    label = LabelEncoder()

    df["Education"] = label.fit_transform(df["Education"])
    df["Property_Area"] = label.fit_transform(df["Property_Area"])

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
