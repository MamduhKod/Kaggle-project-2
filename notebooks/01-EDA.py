import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-2/data/raw/train.csv"
)
print(f"The shape of the train data is {df_train.shape}")

df_train.head()

df_train = df_train.drop("id", axis=1)


for col in df_train.select_dtypes(include=["object"]):
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])


df_train.to_csv(
    "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-2/data/processed/train.csv",
    index=False,
)
