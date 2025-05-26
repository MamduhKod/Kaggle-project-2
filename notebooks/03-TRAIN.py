import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


# load data
X_train = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/X_train.csv"
)
X_cv = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/X_cv.csv"
)
x_test = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/x_test.csv"
)

y_train = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/y_train.csv"
)
y_cv = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/y_cv.csv"
)
y_test = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/y_test.csv"
)


# Define the base model
model = HistGradientBoostingClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 300, 500],
    "max_depth": [4, 6, 8],
    "l2_regularization": [0.0, 0.1, 1.0],
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",  # You can use 'accuracy', 'roc_auc', etc.
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all cores
    verbose=1,
)

# Run the search on your training data
grid_search.fit(X_train, y_train)

# Best model and its parameters
print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# You can now use grid_search.best_estimator_ to predict
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_cv)


from sklearn.metrics import classification_report, confusion_matrix

# Predict on CV set
y_cv_pred = best_model.predict(X_cv)

# Print classification report
print("Cross-validation performance:")
print(classification_report(y_cv, y_cv_pred))

# Predict on test set
y_test_pred = best_model.predict(x_test)

print("Test performance:")
print(classification_report(y_test, y_test_pred))

# Print classification report
print("Cross-validation confusiion matrix performance:")
print(confusion_matrix(y_cv, y_cv_pred))

import joblib

joblib.dump(
    best_model,
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/models/best_histgb_model.pkl",
)
