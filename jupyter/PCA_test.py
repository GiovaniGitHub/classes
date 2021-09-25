import pandas as pd

df_train = pd.read_csv('/home/nobrega/Lets_Code/datasets/train.csv')
df_train.drop(columns=['Id'], inplace=True)

meta_data = {}
for col in df_train.columns:
    if any(df_train[col].isnull()):
        if df_train[col].dtype == float:
            df_train[col] = df_train[col].fillna(df_train[col].mean())
        if df_train[col].dtype == object:
            df_train[col] = df_train[col].fillna('nan')
    if df_train[col].dtype == object:
        meta_data[col] = {value:index for index, value in enumerate(df_train[col].unique())}

df_train.replace(meta_data, inplace= True)