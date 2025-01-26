import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
data = pd.read_csv('student_grades.csv')  

# Remove any leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# Step 2: Explore Dataset
print("Dataset Head:\n", data.head())
print("Dataset Info:\n", data.info())
print("Dataset Description:\n", data.describe())

# Verify columns in the dataset
print("Columns in the dataset:", data.columns)

# Step 3: Data Cleaning
# Handle missing values (if any)
data = data.dropna()

# Step 4: Feature Selection
X = data[['Hours_Studied', 'Previous_Grade', 'Attendance']]  # Ensure column names are correct
y = data['Final_Grade']

# Step 5: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print Coefficients and Intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Step 7: Model Evaluation
# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics Calculation
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training MSE: {mse_train}, MAE: {mae_train}, R2: {r2_train}")
print(f"Testing MSE: {mse_test}, MAE: {mae_test}, R2: {r2_test}")

# Step 8: Residual Analysis
residuals = y_test - y_pred_test

# Residual Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_test, y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Step 9: Visualization
# Scatter Plot with Regression Line
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred_test, line_kws={"color": "red"})
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Residual Distribution')
plt.show()

# Save results
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test,
    'Residuals': residuals
})
results.to_csv('regression_results.csv', index=False)

print("Results saved to 'regression_results.csv'.")
