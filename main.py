import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

missing_values = pd.read_csv("missing_values.csv")