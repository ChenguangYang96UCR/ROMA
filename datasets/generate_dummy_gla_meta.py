import os
import json
import numpy as np
import pandas as pd


root_path = "datasets"
city = "gla"
num_nodes = 200
poi_dim = 768
satellite_dim = 1000
seed = 2036

rng = np.random.default_rng(seed)

os.makedirs(os.path.join(root_path, "poi"), exist_ok=True)
os.makedirs(os.path.join(root_path, "data", city), exist_ok=True)
os.makedirs(os.path.join(root_path, "picture", city), exist_ok=True)


# 1. POI vectors: datasets/poi/gla_poi_1000_vectors.csv
poi_rows = []
for i in range(num_nodes):
    vec = rng.normal(0, 1, poi_dim).astype(float).tolist()
    poi_rows.append({
        "list_id": i,
        "sentence_vector": json.dumps(vec),
    })

poi_path = os.path.join(root_path, "poi", f"{city}_poi_1000_vectors.csv")
pd.DataFrame(poi_rows).to_csv(poi_path, index=False)


# 2. Location JSON: datasets/data/gla/gla.json
loc_data = {}
base_lat = 34.0522
base_lon = -118.2437

for i in range(num_nodes):
    loc_data[str(i)] = {
        "lat": float(base_lat + rng.normal(0, 0.1)),
        "lon": float(base_lon + rng.normal(0, 0.1)),
    }

json_path = os.path.join(root_path, "data", city, f"{city}.json")
with open(json_path, "w") as f:
    json.dump(loc_data, f)


# 3. Satellite/image features: datasets/picture/gla/image_features.csv
sat_rows = []
for i in range(num_nodes):
    vec = rng.normal(0, 1, satellite_dim).astype(float).tolist()
    sat_rows.append({
        "filename": f"{i}.png",
        "feature_vector": json.dumps(vec),
    })

sat_path = os.path.join(root_path, "picture", city, "image_features.csv")
pd.DataFrame(sat_rows).to_csv(sat_path, index=False)


print("written:", poi_path)
print("written:", json_path)
print("written:", sat_path)