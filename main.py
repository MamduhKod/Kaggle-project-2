import os
import sys

# from src.features import feature_engineering, outliers_and_spread, save_data
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    INTERIM_DATA_BASE_PATH,
    HYPERPARAMETERS,
)
from src.dataset import load_data
from src.features import split_data, feature_engineering, outliers_and_spread, save_data
from src.modeling.train_predict import load_training_data
from src.modeling.train_predict import (
    train_model,
    predict_cv,
    predict_test,
    save_trained_model,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def main():
    load_data(RAW_DATA_PATH)

    # Split data
    X_train, X_cv, x_test, y_train, y_cv, y_test = split_data(PROCESSED_DATA_PATH)

    # Feature engineering
    X_train, X_cv, x_test = feature_engineering(X_train, X_cv, x_test)

    # Outliers and spread
    outliers_and_spread(X_train)

    # Save data
    save_data(X_train, X_cv, x_test, y_train, y_cv, y_test)

    # Load training data
    X_train, y_train, X_cv, y_cv, x_test, y_test = load_training_data(
        INTERIM_DATA_BASE_PATH
    )

    # Train model
    model = train_model(X_train, y_train, HYPERPARAMETERS)

    # Predict on CV set
    predict_cv(model, X_cv, y_cv)

    # Predict on test set
    predict_test(model, x_test, y_test)

    # Save trained model
    save_trained_model(model)


if __name__ == "__main__":
    main()
