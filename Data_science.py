    # lets work on deep learning
    # lets do this 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from pathlib import Path
from icecream import ic 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
   
   # lets laad the dataset -->

# Load dataset
data_path = Path('machine_learning_algo.csv')
if data_path.exists():
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f'This file path does not exist: {data_path}')

# Features and target
x = data[['Sleep_Quality', 'Study_Hours']]
y = data['Project_Hours']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Visualize the data
plt.figure(figsize=(6, 8))
sns.scatterplot(x='Sleep_Quality', y='Project_Hours', data=data, label='Sleep Quality vs Project Hours')
sns.scatterplot(x='Study_Hours', y='Project_Hours', data=data, label='Study Hours vs Project Hours', color='orange')
plt.xlabel('Features (Sleep Quality / Study Hours)')
plt.ylabel('Target (Project Hours)')
plt.legend()
plt.grid(True)
plt.show()

# Ridge Regression
ridge = Ridge(alpha=1)
ridge.fit(x_train, y_train)
ridge_accuracy = ridge.score(x_test, y_test) * 100
ridge_pred = ridge.predict(x_test)

# Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_pred)
ic(f'Ridge Accuracy: {ridge_accuracy:.2f}%')
ic(f'Ridge MSE: {ridge_mse:.2f}')

# Linear Regression
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train, y_train)
linear_accuracy = linear.score(x_test, y_test) * 100
linear_pred = linear.predict(x_test)

# Evaluate Linear Regression
linear_mse = mean_squared_error(y_test, linear_pred)
ic(f'Linear Regression Accuracy: {linear_accuracy:.2f}%')
ic(f'Linear Regression MSE: {linear_mse:.2f}')

# Compare Results
print("\n--- Model Comparison ---")
print(f"Ridge Regression Accuracy: {ridge_accuracy:.2f}% | MSE: {ridge_mse:.2f}")
print(f"Linear Regression Accuracy: {linear_accuracy:.2f}% | MSE: {linear_mse:.2f}")



