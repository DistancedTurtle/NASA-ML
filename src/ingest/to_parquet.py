import json
from pathlib import Path

import pandas as pd

data_dir = Path.cwd() / "data"
file_path = data_dir / "near_earth_objects.json"

with open(file_path) as f:
    master_neo_dict = json.load(f)

df = pd.json_normalize(list(master_neo_dict.values()), sep="_")

feature_columns = [
    "absolute_magnitude_h",
    "estimated_diameter_kilometers_estimated_diameter_min",
    "estimated_diameter_kilometers_estimated_diameter_max",
    "is_sentry_object",
    "orbital_data_observations_used",
    "orbital_data_orbit_uncertainty",
    "orbital_data_data_arc_in_days",
    "orbital_data_eccentricity",
    "orbital_data_semi_major_axis",
    "orbital_data_inclination",
    "orbital_data_ascending_node_longitude",
    "orbital_data_orbital_period",
    "orbital_data_perihelion_distance",
    "orbital_data_perihelion_argument",
    "orbital_data_aphelion_distance",
    "orbital_data_mean_anomaly",
    "orbital_data_mean_motion",
    "orbital_data_jupiter_tisserand_invariant",
    "orbital_data_minimum_orbit_intersection",
    "orbital_data_orbit_class_orbit_class_type",
    "orbital_data_orbit_class_orbit_class_description",
    "orbital_data_orbit_class_orbit_class_range",
]

target_column = "is_potentially_hazardous_asteroid"

df_ml = df[feature_columns + [target_column]].copy()

numeric_columns = [
    "absolute_magnitude_h",
    "estimated_diameter_kilometers_estimated_diameter_min",
    "estimated_diameter_kilometers_estimated_diameter_max",
    "orbital_data_observations_used",
    "orbital_data_orbit_uncertainty",
    "orbital_data_data_arc_in_days",
    "orbital_data_eccentricity",
    "orbital_data_semi_major_axis",
    "orbital_data_inclination",
    "orbital_data_ascending_node_longitude",
    "orbital_data_orbital_period",
    "orbital_data_perihelion_distance",
    "orbital_data_perihelion_argument",
    "orbital_data_aphelion_distance",
    "orbital_data_mean_anomaly",
    "orbital_data_mean_motion",
    "orbital_data_jupiter_tisserand_invariant",
    "orbital_data_minimum_orbit_intersection",
]
df_ml[numeric_columns] = df_ml[numeric_columns].apply(pd.to_numeric, errors="coerce")

output_path = data_dir / "neos_ml.parquet"
df_ml.to_parquet(output_path)
print(f"Wrote {len(df_ml)} rows to {output_path}")
