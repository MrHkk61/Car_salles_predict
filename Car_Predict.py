 # Data
import numpy as np
import pandas as pd

# Data Visualization
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Data Pre-Processing
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor

# Model Performance
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import metrics
df = pd.read_csv("dataset.csv")
org_df = df.copy()

# Display
df.head()
df.info()

numerical_cols, categorical_cols = [], []

# Identify columns
for col in df.columns:
    if df[col].dtype=="int64":
        numerical_cols.append(col)
    elif df[col].dtype=="float64":
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)

# Record 
n_numerical = len(numerical_cols)
n_categorical = len(categorical_cols)

# Show
print("Toplam Sayısal Sütun Sayısı   : {}".format(n_numerical))
print("Toplam Kategorik Sütun Sayısı :: {}".format(n_categorical))

df[categorical_cols[0]]

df.drop(columns=['CarName', 'car_ID'], inplace=True)

numerical_cols.remove('car_ID')
categorical_cols.remove('CarName')

fueltype_mapping = {'gas':0.0, 'diesel':1.0}
df[categorical_cols[0]] = df[categorical_cols[0]].map(fueltype_mapping)
df[categorical_cols[0]].unique()

aspiration_mapping= {'std':0.0, 'turbo':1.0}
df[categorical_cols[1]] = df[categorical_cols[1]].map(aspiration_mapping)
df[categorical_cols[1]].unique()


df[categorical_cols[2]].unique()
doornumber_mapping= {'two':2.0, 'four':4.0}
df[categorical_cols[2]] = df[categorical_cols[2]].map(doornumber_mapping)
df[categorical_cols[2]].unique()
df[categorical_cols[3]].unique()

encoder = OrdinalEncoder()
df[categorical_cols[3]] = encoder.fit_transform(df[categorical_cols[3]].to_numpy().reshape(-1,1))

df[categorical_cols[3]].unique()

carbody_cats = encoder.categories_[0]
carbody_cats

df[categorical_cols[4]].unique()

drivewheel_mapping = {'rwd':0.0, 'fwd':1.0, '4wd':2.0}
df[categorical_cols[4]] = df[categorical_cols[4]].map(drivewheel_mapping)

enginelocation_mapping = {'front':0.0, 'rear':1.0}
df[categorical_cols[5]] = df[categorical_cols[5]].map(enginelocation_mapping)

encoder = OrdinalEncoder()
df[categorical_cols[6]] = encoder.fit_transform(df[categorical_cols[6]].to_numpy().reshape(-1,1))
df[categorical_cols[6]].unique()

encoder = OrdinalEncoder()
df[categorical_cols[7]] = encoder.fit_transform(df[categorical_cols[7]].to_numpy().reshape(-1,1))

df[categorical_cols[7]].unique()

encoder = OrdinalEncoder()
df[categorical_cols[8]] = encoder.fit_transform(df[categorical_cols[8]].to_numpy().reshape(-1,1))

df[categorical_cols[8]].unique()

df.info()

y_full = df.pop('price')
X_full = df


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_full)



X_full.drop(columns=['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'], inplace=True)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y_full)

lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction
pred = lr.predict(X_valid)

# Performance Measure
lr_mse = mean_squared_error(y_valid, pred)
lr_r2 = r2_score(y_valid, pred)

# Show Measures
result = '''
Ortalama Kare Hatası  : {}
Model Performansı  : {}
'''.format(lr_mse, lr_r2)

print(result)
















