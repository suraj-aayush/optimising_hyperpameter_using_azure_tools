import argparse
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def main(args):
    mlflow.autolog()
    # Read data
    df = get_data(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    model = train_model(args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf, X_train, X_test, y_train, y_test)

    # Evaluate model
    eval_model(model, X_test, y_test)

# Function to read data
def get_data(path):
    print("Reading data...")
    df = pd.read_csv(path)
    return df

# Function to split data
def split_data(df):
    print("Splitting data...")
    X = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values
    y = df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test

# Function to train model
def train_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, X_train, X_test, y_train, y_test):
    print("Training model...")
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

# Function to evaluate model
def eval_model(model, X_test, y_test):
    # Evaluate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)

    # Evaluate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:,1])
    print('AUC:', auc)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--n_estimators", dest='n_estimators', type=int, nargs='+', default=[100])
    parser.add_argument("--max_depth", dest='max_depth', type=int, nargs='+', default=[None])
    parser.add_argument("--min_samples_split", dest='min_samples_split', type=int, nargs='+', default=[2])
    parser.add_argument("--min_samples_leaf", dest='min_samples_leaf', type=int, nargs='+', default=[1])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60)
    print("\n\n")
