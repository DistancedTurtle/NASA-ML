import json
from pathlib import Path

import pandas as pd


json_file_path = Path.cwd() / "data" / "near_earth_objects.json"

with open(json_file_path) as f:
    master_neo_dict = json.load(f)

df = pd.json_normalize(list(master_neo_dict.values()), sep="_")

# Note: orbital_data_orbit_class_orbit_class_range is dropped, not parsed.
# It's a textual *definition* of the orbit class (constant per class, and
# different physical quantities per class), so any numbers extracted from it
# are fully redundant with orbit_class_type (one-hot encoded below) and the
# real per-asteroid orbital_data_perihelion_distance feature.

dropped_cols = [
    'orbital_data_orbit_class_orbit_class_range', 
    'id', 
    'name', 
    'designation', 
    'neo_reference_id', 
    'nasa_jpl_url', 
    'links_self', 
    'close_approach_data', 
    'orbital_data_orbit_determination_date', 
    'orbital_data_first_observation_date', 
    'orbital_data_last_observation_date', 
    'orbital_data_orbit_id', 
    'orbital_data_equinox', 
    'name_limited',
    'sentry_data',
    'orbital_data_orbit_class_orbit_class_description',
    # --- Dropped to de-leak the target. A PHA is *defined* by the IAU as
    # MOID <= 0.05 AU AND H <= 22.0, so these directly compute the label:
    'orbital_data_minimum_orbit_intersection',  # the MOID half
    'absolute_magnitude_h',                      # the H (size) half
    # estimated_diameter is computed from H (D ~ 10^(-H/5)), so every unit
    # variant is just H in disguise -> also leaks the size half:
    'estimated_diameter_kilometers_estimated_diameter_min',
    'estimated_diameter_kilometers_estimated_diameter_max',
    'estimated_diameter_meters_estimated_diameter_min',
    'estimated_diameter_meters_estimated_diameter_max',
    'estimated_diameter_miles_estimated_diameter_min',
    'estimated_diameter_miles_estimated_diameter_max',
    'estimated_diameter_feet_estimated_diameter_min',
    'estimated_diameter_feet_estimated_diameter_max',
]

df = df.drop(columns=dropped_cols, errors='ignore')

bool_cols = ['is_potentially_hazardous_asteroid', 'is_sentry_object']
df[bool_cols] = df[bool_cols].astype(int)

numeric_cols = [
    'orbital_data_data_arc_in_days',  # already float
    'orbital_data_observations_used',  # already float
    'orbital_data_orbit_uncertainty',
    'orbital_data_jupiter_tisserand_invariant',
    'orbital_data_epoch_osculation',
    'orbital_data_eccentricity',
    'orbital_data_semi_major_axis',
    'orbital_data_inclination',
    'orbital_data_ascending_node_longitude',
    'orbital_data_orbital_period',
    'orbital_data_perihelion_distance',
    'orbital_data_perihelion_argument',
    'orbital_data_aphelion_distance',
    'orbital_data_perihelion_time',
    'orbital_data_mean_anomaly',
    'orbital_data_mean_motion',]
df[numeric_cols] = df[numeric_cols].astype(float)


df = pd.get_dummies(df, columns=df.select_dtypes(include=['object','str','category']).columns, dummy_na=True, dtype=int)



output_path =  Path.cwd() / "data" / "neos_ml.parquet"
df.to_parquet(output_path)
print(f"Wrote {len(df)} rows to {output_path}")
