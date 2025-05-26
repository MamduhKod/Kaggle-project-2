import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.config import INTERIM_DATA_BASE_PATH, HYPERPARAMETERS
import joblib


def load_training_data(data_path):
    X_train = pd.read_csv(data_path + "X_train.csv")
    X_cv = pd.read_csv(data_path + "X_cv.csv")
    x_test = pd.read_csv(data_path + "x_test.csv")

    y_train = pd.read_csv(data_path + "y_train.csv")
    y_cv = pd.read_csv(data_path + "y_cv.csv")
    y_test = pd.read_csv(data_path + "y_test.csv")

    return X_train, y_train, X_cv, y_cv, x_test, y_test


X_train, y_train, X_cv, y_cv, x_test, y_test = load_training_data(
    INTERIM_DATA_BASE_PATH
)


def train_model(X_train, y_train, hyperparameters):
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=hyperparameters["max_iter"],
        learning_rate=hyperparameters["learning_rate"],
        l2_regularization=hyperparameters["l2_regularization"],
        max_depth=hyperparameters["max_depth"],
    )
    model.fit(X_train, y_train)
    return model


# Train model based on best parameters
model = train_model(X_train, y_train, HYPERPARAMETERS)


def predict_cv(model, X_cv, y_cv):
    # Predict on CV set
    y_cv_pred = model.predict(X_cv)

    # Print classification report
    print("Cross-validation performance:")
    print(classification_report(y_cv, y_cv_pred))

    # Print classification report
    print("Cross-validation confusiion matrix performance:")
    print(confusion_matrix(y_cv, y_cv_pred))


y_cv_pred = predict_cv(model, X_cv, y_cv)


def predict_test(model, x_test, x_cv):
    # Predict on test set
    y_test_pred = model.predict(x_test)

    print("Test performance:")
    print(classification_report(y_test, y_test_pred))

    # Print classification report
    print("Text confusiion matrix performance:")
    print(confusion_matrix(y_cv, y_cv_pred))


y_test_pred = predict_cv(model, x_test, y_test)


def save_trained_model(model):
    joblib.dump(
        model,
        "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/models/best_histgb_model.pkl",
    )
