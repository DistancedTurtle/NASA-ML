from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json


file_path = Path.cwd() / "data" / "neos_ml.parquet"
df = pd.read_parquet(file_path)
pd.set_option('display.max_columns', None)
print(df.head())
all_values = df.values
tensor_data = torch.tensor(all_values, dtype = torch.float32)
print(tensor_data[0,:])