import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import f_regression

x=np.loadtxt("ex3x.dat")
y=np.loadtxt("ex3y.dat")

y2=y.reshape(-1,1)

polynomial_features= PolynomialFeatures(degree= 3)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

mpl.rcParams['legend.fontsize'] = 12
fig = plt.figure()
ax = fig.gca(projection ='3d')
ax.scatter(x[:, 0], x[:, 1], y, label ='y', s = 5,color ="dodgerblue")
ax.scatter(x[:, 0], x[:, 1], y_poly_pred, label ='y_pred', s = 5,color ="orange")
ax.legend()
ax.view_init(45, 0)

plt.show()