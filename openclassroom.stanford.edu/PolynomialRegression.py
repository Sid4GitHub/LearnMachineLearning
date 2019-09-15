import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import f_regression


y = np.array([984.5410628019, 128.8194444444, 94.495412844, 312.0331950207,
65.1127819549, 168.3289588801, 301.7441860465, 90.1408450704,
249.4573643411, 239.0361445783, 181.1775200714, 327.2440944882,
230.9523809524, 158.9442815249, 30.3759398496, 152.5783619818,
157.5938566553, 150.9933774834, 92.2413793103, 37.5706214689,
161.2958226769, 125.1546391753, 181.0394610202, 423.9678899083,
449.3975903614, 208.9949518128, 151.5695067265, 205.4315027158,
187.1388998813, 121.733615222, 155.1660516605, 196.2901896125,
211.9626168224, 120.3170028818, 104.2812254517, 245.7815565729,
133.5061088486, 135.6054191363, 102.4313372355, 148.6814566764,
125.9274357929, 166.783352781, 104.2042042042, 203.8586703859,
183.3114323259, 251.4944939696, 124.5541022592, 264.9880095923])

x = np.array([0.09, 0.12, 0.17, 0.23, 0.31, 0.32, 0.33, 0.34, 0.38, 0.42, 0.43,
0.5, 0.51, 0.52, 0.55, 0.74, 0.78, 0.79, 0.8, 0.83, 0.84, 0.93, 0.95,
1.05, 1.05, 1.09, 1.11, 1.13, 1.2, 1.21, 1.24, 1.27, 1.33, 1.35, 1.38,
1.43, 1.45, 1.45, 1.48, 1.6, 1.62, 1.7, 1.76, 1.79, 1.79, 1.79, 1.94, 2])

x2=x[:,np.newaxis]
y2=y.reshape(-1,1)

polynomial_features= PolynomialFeatures(degree= 4)
x_poly = polynomial_features.fit_transform(x2)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, y_poly_pred, 'b', label='Fitted line')
plt.show()