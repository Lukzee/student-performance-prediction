import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import io

df = pd.read_csv('student_performance.csv')
# print(df.head())

df['extra_curricular'] = df['extra_curricular'].map({'Yes': 1, 'No': 0})

# Feature matrix and target
X = df.drop(columns=['student_id', 'final_grade'])
y = df['final_grade']

# Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
# model.fit(X_train, y_train)
model.fit(X, y)

joblib.dump(model, 'student_performance.joblib')

# Predictions
# y_pred = model.predict(X_test)

# print(y_pred)