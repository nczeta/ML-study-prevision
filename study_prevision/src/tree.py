# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from functions import *

# Fetching the Dataset
df = pd.read_csv('../data/dataset.csv')

# One hot Encoding
df_encoded = pd.get_dummies(df, columns=['subject', 'study'], drop_first=True)

# Setting the target and the features
y = df_encoded['preparation']
x = df_encoded.drop(columns=['preparation'])

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Finding the better depth
check = float('inf')
index = 0
for i in range(1, 20):
    model = DecisionTreeRegressor(max_depth=i)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    mae = mean_absolute_error(y_test, prediction)
    if mae < check:
        check = mae
        index = i

print(f"The best depth is {index}")

# Depth equal to 11
mod = DecisionTreeRegressor(max_depth=11)
mod.fit(x_train, y_train)

# Prediction
pred = mod.predict(x_test)
clip_pred = np.clip(pred, a_min=0, a_max=10)

# Checking the first five predictions
for i in range(5):
    print(f"The prediction is {clip_pred[i]}, the real target is {y_test.iloc[i]}")

# Evalueting the model
e, mse, mae = evaluating_model(pred, y_test)

# Predict your data
data = get_data(x_train)

# Calculating the prediction of your data
my_prediction = predict_data(data, mod)

# Output
print("You will receive ", my_prediction)