import pandas as pd
import pickle
import argparse

def preprocess_test_data(file_path, training_columns):
    df = pd.read_csv(file_path)

    # Handle missing values for numeric columns by filling with the median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Handle missing values for categorical columns by filling with the most frequent value (mode)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Convert categorical columns to numerical using one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Align the test set to have the same columns as the training set
    df = df.reindex(columns=training_columns, fill_value=0)

    return df

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(input_file, model_path):
    # Load the model
    model = load_model(model_path)

    # Get the training columns from the model
    training_columns = model.feature_names_in_

    # Preprocess the test data
    X_test = preprocess_test_data(input_file, training_columns)

    # Make predictions
    predictions = model.predict(X_test)

    # Load test file to get the IDs (or other relevant columns)
    df_test = pd.read_csv(input_file)
    df_test['SalePrice_Prediction'] = predictions

    # Save the predictions to a CSV file
    df_test[['Id', 'SalePrice_Prediction']].to_csv('predictions.csv', index=False)
    print('Predictions saved to predictions.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/test.csv', help='Path to the input CSV file containing test data')
    parser.add_argument('--model', default='./models/linear_regression_model.pkl', help='Path to the saved model')
    args = parser.parse_args()

    predict(args.input, args.model)
