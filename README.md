# Credit Card Default Prediction

This repository contains a machine learning project for predicting whether a credit card user is likely to default on the next payment.

## Project Overview

The project uses a real-world credit card dataset and trains a classification model to identify customers with a high risk of default. A Streamlit web app is also included to make predictions interactively.

## Repository Structure

- `UCI_Credit_Card.csv` - Dataset used for model training and evaluation
- `model.py` - Data preprocessing, model comparison, hyperparameter tuning, and final model training
- `app.py` - Streamlit application for user login, prediction, and analysis
- `catboost_model.pkl` - Trained model saved for prediction use
- `requirements.txt` - Python dependencies required to run the project
- `.devcontainer/` - Development container configuration
- `catboost_info/` - CatBoost training metadata

## Dataset

The dataset is the UCI Credit Card dataset, which contains customer financial information and repayment behavior features. The target variable is:

- `default.payment.next.month`

## Objective

The goal of this project is to build a robust predictive model that helps estimate whether a customer is likely to default based on their financial profile and repayment history.

## Models Evaluated

The project compares multiple machine learning models, including:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

The best-performing model is saved and used in the web application.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- Streamlit

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

This launches the web interface where you can:

- create an account
- log in
- enter customer details
- predict default risk
- view analysis of saved CSV results

## Model Training

To retrain the model, run:

```bash
python model.py
```

This script loads the dataset, cleans it, tunes models, evaluates performance, and saves the final trained model as `catboost_model.pkl`.

## Prediction Output

The model predicts whether a customer is likely to default or not and also provides probabilities for:

- Not Default
- Default

## Notes

This project is intended for learning, experimentation, and demonstration of a credit risk prediction workflow in a real-world scenario.
