import pandas as pd
from config import RAW_DATA_PATH
from sklearn.preprocessing import LabelEncoder


def load_data(data_path):
    df_train = pd.read_csv(data_path)
    print(f"The shape of the train data is {df_train.shape}")

    df_train.head()

    df_train = df_train.drop("id", axis=1)

    for col in df_train.select_dtypes(include=["object"]):
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])

    return df_train


try:
    load_data(RAW_DATA_PATH).to_csv(
        "/Users/mamduhhalawa/Desktop/Mlrepos/kaggle-project-2/data/processed/train.csv",
        index=False,
    )
except Exception as e:
    print("Something went wrong with data loading:", str(e))
else:
    print("Successfully added processed data.")
