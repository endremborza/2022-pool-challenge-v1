from pathlib import Path
import json
import pandas as pd

if __name__ == "__main__":
    with open('treesave.pkl', 'rb') as handle:
        tree = pickle.load(handle)

    input_locations = json.loads(Path("input.json").read_text())

    input = np.array([list(i.values()) for i in input_locations])

    dist, ind = tree.query(input, k = 5)

    distances = [dist[i] for i in range(len(dist))]

    indexes = []
    for i in range(len(distances)):
        closest_index = np.where(distances[i] == np.amin(distances[i]))[0][0]
        closest_index = ind[i][closest_index]
        indexes.append(closest_index)

    data = pd.read_pickle("data.pkl")
    own_results = data.iloc[indexes,[1,-2,-1]].to_dict("records")
    Path("output.json").write_text(json.dumps(own_results))