# Study Prevision

A machine learning project that predicts your exam grade based on your study habits.

## Description

Given a few inputs, hours studied, stress level, subject, and study method, the model predicts a grade between 0 and 10. Two different models are compared: a **Linear Regression** and a **Decision Tree Regressor**, with the linear regression performing slightly better.

## Project Structure

```
study_prevision/
│
├── data/
│   └── dataset.csv               # Dataset used for training
│
├── src/
│   ├── functions.py              # Utility functions (data loading, evaluation, prediction)
│   ├── regression.py             # Linear Regression model
│   └── tree.py                   # Decision Tree model
│
├── notebooks/
│   ├── data_exploration.ipynb    # Dataset exploration and feature analysis
│   └── comparison.ipynb          # Model comparison and error analysis
│
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/nczeta/ML-study-prevision.git
cd study_prevision
pip install -r requirements.txt
```

## Usage

Run the Linear Regression model:
```bash
python src/regression.py
```

Run the Decision Tree model:
```bash
python src/tree.py
```

Both scripts will ask for the following inputs:

| Input | Description | Example |
|---|---|---|
| Hours studied | How many hours you studied | `3.5` |
| Stress level | Your stress level (numeric) | `2.0` |
| Subject | One of: history, philosophy, computer_science, italian, art, math, physics, english, science | `math` |
| Study method | One of: read, flashcards, repeat, exercise | `flashcards` |

## Results

| Model | ME | MAE | MSE |
|---|---|---|---|
| Linear Regression | -0.030 | 0.412 | 0.267 |
| Decision Tree | 0.028 | 0.488 | 0.388 |

The Linear Regression model performs better across all metrics.
