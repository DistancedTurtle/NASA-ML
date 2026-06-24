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

df = pd.read_json(f'{Path.cwd()}/data/near_earth_objects.json')
df.info()
