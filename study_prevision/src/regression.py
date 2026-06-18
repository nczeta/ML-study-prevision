#Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from functions import *

#Fetching the Dataset
df = fetch_dataset()

# One hot encoding
df_encode = pd.get_dummies(df, columns=['subject', 'study'], drop_first=True)

# Target and Features
y = df_encode['preparation']
x = df_encode.drop(columns=['preparation'])

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()

# Training the model
model.fit(x_train, y_train)

# Predicting
prediction = model.predict(x_test)

# Applying a min e max value
refined_prediction = np.clip(prediction, a_min=0, a_max=10)

# Small Comparison
for i in range(5):
    print(f"Number {i} - Predicted value: {refined_prediction[i]}, Real value: {y_test.iloc[i]}")


# Evalueting the model
e, mse, mae = evaluating_model(prediction, y_test)

# Predict your data
my_data = get_data(x_train)

# Calculating the prediction of your data
my_prediction = predict_data(my_data, model)

# Output
print("You will receive ", my_prediction)