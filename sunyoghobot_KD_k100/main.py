import pandas as pd
from pathlib import Path
from math import radians
from sklearn.neighbors import KDTree
import numpy as np
import json

if __name__ == "__main__":
    input_locations = json.loads(Path("input.json").read_text())
    data = pd.read_csv("data.csv")
    data = data.loc[data["keycode"] == "p", :].reset_index()
    swimmers = [(data["x_position"][i],data["y_position"][i],data["z_position"][i]) for i in range(len(data))]
    objects = [(input_locations[i]["x_position"],input_locations[i]["y_position"],input_locations[i]["z_position"]) for i in range(len(input_locations))]
    swimmers_rad = np.array([[radians(x[0]), radians(x[1]), radians(x[2])] for x in swimmers ])
    objects_rad = np.array([[radians(x[0]), radians(x[1]), radians(x[2])] for x in objects ])
    
    tree = KDTree(swimmers_rad)
    dist, ind = tree.query(objects_rad, k = 100)
    
    results_index = [min([i for d, i in zip(d_vec, i_vec) if d == d_vec.min()]) for d_vec, i_vec in zip(dist, ind)]
    solutions = data.iloc[results_index,:][["msec", "subject", "trial"]]
    solutions["msec"] = solutions["msec"].astype("object")
    solutions["trial"] = solutions["trial"].astype("object")
    
    out = ([solutions.iloc[i].to_dict() for i in range(len(solutions))])
    Path("output.json").write_text(json.dumps(out))