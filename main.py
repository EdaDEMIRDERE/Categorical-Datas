import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

missing_values = pd.read_csv("missing_values.csv")
print(missing_values)

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
age = missing_values.iloc[:, 1:4].values
print(age)

imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])
print(age)