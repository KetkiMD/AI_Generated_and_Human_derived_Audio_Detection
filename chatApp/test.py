import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from joblib import dump, load

import warnings
warnings.simplefilter("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler

data=pd.read_csv("Features_final.csv")

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

print("Y : ",len(y))

scaler = StandardScaler()

X = np.array(data.iloc[:, :-1], dtype = float)
print("X : ",len(X))


lin_regressor = LinearRegression()
lin_regressor.fit(X,y)
dump(lin_regressor, 'NN.H5')

print("Done")
