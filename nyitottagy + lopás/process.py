import pandas as pd
from pathlib import Path
import scipy.spatial.distance
import numpy as np
import json
from sklearn.neighbors import KDTree
import pickle


if __name__ == "__main__":


    data = pd.read_pickle("data.pkl")

    if data.shape[0] > 9_000_000:

        data = pd.read_parquet("data_subset.parquet")

        input_locations = json.loads(Path("input.json").read_text())
        input_locations = np.array(
            [tuple(pos.values()) for pos in input_locations], dtype=np.float64
        )

        with open("kdtree.pickle", "rb") as file:
            tree = pickle.load(file)

        dist, ind = tree.query(input_locations, k = 100)

        results_index = [min([i for d, i in zip(d_vec, i_vec) if d == d_vec.min()]) for d_vec, i_vec in zip(dist, ind)]
        results = data.iloc[results_index, :][["msec", "subject", "trial"]].to_dict("records")
        Path("output.json").write_text(json.dumps(results))



    else:

        df = data.reset_index().loc[:,["x_position","y_position","z_position"]]

        input_locations = json.loads(Path("input.json").read_text())

        input = np.array([list(i.values()) for i in input_locations])

        d = scipy.spatial.distance.cdist(list(df.values), np.array(input))

        shortest_index = [d.T[i].argmin() for i in range(len(d.T))]

        result = data.iloc[shortest_index, [1, -2, -1]].to_dict("records")
        Path("output.json").write_text(json.dumps(result))