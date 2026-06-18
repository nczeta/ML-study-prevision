from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

def fetch_dataset():
    path = '../data/dataset.csv'
    df = pd.read_csv(path)
    return df

def evaluating_model(predict, y_test):
    e = predict - y_test
    e = e.mean()
    mse = mean_squared_error(y_test, predict)
    mae = mean_absolute_error(y_test, predict)

    print("The mean error is", e)
    print("The mean squared error is", mse)
    print("The mean absolute error is", mae)

    return e, mse, mae

def get_data(x_train):
    # Defining constant
    subjects_list = [
        'history', 'philosophy', 'computer_science',
        'italian', 'art', 'math', 'physics', 'english', 'science'
        ]

    study_methods = [
        'read', 'flashcards', 'repeat', 'exercise' 
        ]

    # Asking for the inputs  
    hours = float(input("Insert how many hours you have studied: "))
    stress = float(input("Insert how stressed you are: "))

    while(True):
        subject = input("Insert the subject: ")
        if subject not in subjects_list:
            continue
        else:
            break

    while(True):
        method = input("Insert the study method: ")
        if method not in study_methods:
            continue
        else:
            break

    # Creating the data
    my_data = pd.DataFrame([[hours, stress, subject, method]], columns=['hours', 'stress', 'subject', 'study'])
    my_data = pd.get_dummies(my_data, columns=['subject', 'study'], drop_first=True)
    my_data = my_data.reindex(columns=x_train.columns, fill_value=0)

    return my_data

def predict_data(data, model):
    my_pred = model.predict(data)
    my_pred = np.clip(my_pred, a_min=0, a_max=10)
    return my_pred
        