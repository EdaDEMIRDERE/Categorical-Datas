import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

missing_values = pd.read_csv("missing_values.csv")
print(missing_values)

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
age = missing_values.iloc[:, 1:4].values
print(age)

imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])
print(age)

country = missing_values.iloc[:, 0:1].values
print(country)

label_encoder = preprocessing.LabelEncoder()
country[:, 0] = label_encoder.fit_transform(missing_values.iloc[:, 0])
print(country)

one_hot_encoder = preprocessing.OneHotEncoder()
country = one_hot_encoder.fit_transform(country).toarray()
print(country)

print(list(range(22)))
result_1 = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "us"])
print(result_1)

result_2 = pd.DataFrame(data=age, index=range(22), columns=["boy", "kilo", "yas"])
print(result_2)

gender = missing_values.iloc[:, -1].values
print(gender)

result_3 = pd.DataFrame(data=gender, index=range(22), columns=["gender"])
print(result_3)

s_1 = pd.concat([result_1, result_2], axis=1)
print(s_1)

s_2 = pd.concat([s_1, result_3], axis=1)
print(s_2)