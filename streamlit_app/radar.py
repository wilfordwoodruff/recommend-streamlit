import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler


id_map = pd\
    .read_csv("derived_data.csv")\
    .set_index('UUID')['Internal ID']\
    .to_dict()

# CLEAN HUME OUTPUT
humedata = pd\
    .read_csv("uuid-hume.csv")\
    .replace(id_map)\
    .rename(columns={'UUID': 'internal_id'})\
    .sort_values(by='internal_id')  

emotion = pd\
    .read_csv("emotion_categories.csv")\
    .set_index('HUME')["10CATEGORIES"]\
    .to_dict()

columns_to_normalize = [col for col in humedata.columns if col != 'internal_id']

for column in columns_to_normalize:
    min_col = humedata[column].min()
    max_col = humedata[column].max()
    humedata[column] = (humedata[column] - min_col) / (max_col - min_col)

ids = list(humedata["internal_id"])

ids.insert(0, "emotions")

humeT = humedata.reset_index(drop=True).T.reset_index()

humeT.columns = ids

keep_emotions = list(set(emotion.values()))

order = [
            "Joy-Gratitude",
            "Excitement",
            "Neutral",
            "Anger",
            "Sad",
            "Fear",
            "Satisfaction",
            "Hope",
        ]

emotions_df = humeT\
    .assign(emotions = lambda x: x["emotions"].map(emotion))\
    .query("emotions in @keep_emotions")\
    .groupby("emotions", as_index=False)\
    .mean()\
    .assign(emotions = lambda x: pd.Categorical(x["emotions"], categories=order, ordered=True))\
    .sort_values("emotions")\
    .set_index("emotions")

scaler = MinMaxScaler()
emotions_df = pd.DataFrame(scaler.fit_transform(emotions_df.T).T, columns=emotions_df.columns)

emotions_df = emotions_df.set_index(pd.Index(order))

emotions_df.loc["Satisfaction"] = emotions_df.loc["Satisfaction"]*0.75

emotions_df.to_csv("emotions.csv")
