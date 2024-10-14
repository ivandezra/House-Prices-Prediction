from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from data_processing import preprocess_data

# Preprocess the data
X, y = preprocess_data('./data/train.csv')

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('./models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Validate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
print(f'Validation RMSE: {rmse:.2f}')
