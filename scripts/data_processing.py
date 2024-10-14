import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values for numeric columns by filling with the median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Handle missing values for categorical columns by filling with the most frequent value (mode)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Convert categorical columns to numerical using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Separate features and target variable
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    return X, y

if __name__ == "__main__":
    X, y = preprocess_data('./data/train.csv')
    print(X.head())
