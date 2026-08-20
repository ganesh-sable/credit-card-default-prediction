import numpy as np
import pandas as pd

df = pd.read_csv("UCI_Credit_Card.csv")
print(df.head())

print("\nNull values: ")
print(df.isnull().sum())

print("\nInfo: ")
print(df.info())

print("\nData Shape: ")
print(df.shape)

print("\nDuplicate rows count:", df.duplicated().sum())

df = df.drop_duplicates()
print("After Drop Duplicate rows: ")
print(df.shape)

X = df.drop(columns =["ID", "default.payment.next.month"])
y = df["default.payment.next.month"]

from sklearn.model_selection import train_test_split, RandomizedSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42,
    stratify= y
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



models ={
    "RandomForest" : Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy= "median")),
            ("model", RandomForestClassifier(class_weight= "balanced"))
        ]
    ),

    "XGBoost": Pipeline(
        steps= [
            ("imputer", SimpleImputer(strategy= "median")),
            ("model", XGBClassifier(eval_metric="logloss"))
        ]
    ),

    "LightGBM": Pipeline(
        steps= [
            ("imputer", SimpleImputer(strategy= "median")),
            ("model", LGBMClassifier(class_weight= "balanced"))
        ]
    ),

    "CatBoost": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy= "median")),
            ("model", CatBoostClassifier(verbose=0, scale_pos_weight=3.52, random_state=42))
        ]
    )
}

cat_models = []
model_names = []
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    #print(f"\n{model} Accuracy: ", accuracy_score(y_test, y_pred))
    cat_models.append(acc)
    model_names.append(name)

result = pd.DataFrame(
        {
            "Model Name": model_names,
            "Accuracy" : cat_models      
    }
)

print()
print(result)

param_grids = {
    "RandomForest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    },

    "XGBoost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0]
    },

    "LightGBM": {
        "model__n_estimators": [100, 200, 300],
        "model__num_leaves": [15, 31, 50],
        "model__max_depth": [-1, 5, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0]
    },

    "CatBoost": {
        "model__iterations": [100, 200, 300],
        "model__depth": [4, 6, 8, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__l2_leaf_reg": [1, 3, 5, 7]
    }
}

results = []

best_models = {}

for name, model in models.items():
    random_search = RandomizedSearchCV(
        estimator= model,
        param_distributions= param_grids[name],
        n_iter=10,
        cv= 5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_models[name] = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
            "Model": name,
            "CV Score": random_search.best_score_,
            "Accuracy": accuracy,
            "F1" : f1,
            "Recall": recall,
            "Precision": precision,
            "ROC-AUC": auc,
            "Confusion Matrix": cm
        })

results_df = pd.DataFrame(results)

print("\nFinal Results:")
print(results_df)

final_model = best_models["CatBoost"]

import joblib
joblib.dump(final_model,"catboost_model.pkl")