import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/processed/train.csv"
)

# Separate target and features

y = df_train["loan_status"]
X = df_train.drop("loan_status", axis=1)

# Split data for cross validation
X_train, x_, y_train, y_ = train_test_split(
    X,
    y,
    test_size=0.3,  # 20% test set, 80% train set
    random_state=42,  # for reproducibility
    stratify=y,  # to keep the class distribution same (optional)
)

# Split again for training set
X_cv, x_test, y_cv, y_test = train_test_split(
    x_,
    y_,
    test_size=0.5,
    random_state=42,
)

del x_, y_

print(f"X_train shape: {X_train.shape}")
print(f"X_cv shape: {X_cv.shape}")
print(f"X_test shape: {x_test.shape}")


# spread
for feature in X_train.columns:
    unique = X_train[feature].nunique()
    print(f"{feature}:", unique)

# plot to see outliers
for feature in X_train.columns:
    plt.figure(figsize=(12, 6))
    plt.title(f"{feature}")
    X_train[feature].hist(bins=50)


# Merge to one feature

X_train["person_age_by_income"] = X_train["person_age"] / X_train["person_income"]
X_cv["person_age_by_income"] = X_cv["person_age"] / X_cv["person_income"]
x_test["person_age_by_income"] = x_test["person_age"] / x_test["person_income"]

X_train = X_train.drop(columns=["person_age", "person_income"])
X_cv = X_cv.drop(columns=["person_age", "person_income"])
x_test = x_test.drop(columns=["person_age", "person_income"])


# Calculate correlation matrix
corr = X_train.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True, linewidths=0.5
)
plt.title("Feature Correlation Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Drop highly correlated feature loan percent income (with loan_int_rate)
X_train = X_train.drop(columns=["loan_grade"])
X_cv = X_cv.drop(columns=["loan_grade"])
x_test = x_test.drop(columns=["loan_grade"])

X_train.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/X_train.csv"
)
X_cv.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/X_cv.csv"
)
x_test.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/x_test.csv"
)

y_train.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/y_train.csv"
)
y_cv.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/y_cv.csv"
)
y_train.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/Kaggle-project-2/data/interim/y_test.csv"
)
