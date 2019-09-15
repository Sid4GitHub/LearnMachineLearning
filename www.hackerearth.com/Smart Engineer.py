#https://www.hackerearth.com/problem/machine-learning/smart-engineer/
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import f_regression

xy=np.loadtxt("../data-set/Electricity_Production/test_data.csv",skiprows=1,delimiter=',')
x=xy[:,0:3]
y=xy[:,3]

polynomial_features= PolynomialFeatures(degree= 14)
x_poly = polynomial_features.fit_transform(x)

print(x_poly)

regressor = LinearRegression()
regressor.fit(x_poly, y)

y_pred=regressor.predict(x_poly)

r_squared= regressor.score(x_poly, y)
adjusted_r_squared=1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1)

print("r_squared= "+r_squared.__str__()+"\nadjusted_r_squared="+adjusted_r_squared.__str__())
print("mean_squared_error: "+mean_squared_error(y,y_pred).__str__())

