import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import BallTree


data = pd.read_csv("data.csv")
data = data.loc[data["entity_id"] == 0].drop_duplicates(subset = ["msec","subject","trial"]).dropna().reset_index()
data.to_pickle("data.pkl")

df = data.loc[:,["x_position","y_position","z_position"]]
df_array = np.array(df)
tree = BallTree(df_array, leaf_size = 50, metric = "euclidean")

with open("treesave.pkl", "wb") as handle:
    pickle.dump(tree,handle)