import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score

x=np.loadtxt("ex3x.dat")
y=np.loadtxt("ex3y.dat")
regressor = LinearRegression()
regressor.fit(x, y)

print("Lm="+str(regressor.coef_))
print("Lc="+str(regressor.intercept_))

y_pred = regressor.predict(x)

print(f_regression(x,y))
print(f_regression(x,y_pred))

'''
SS_Residual = sum((y-y_pred)**2)
SS_Total = sum((y-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
'''

r_squared= regressor.score(x, y)
adjusted_r_squared=1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1)

print("r_squared= "+r_squared.__str__()+"\nadjusted_r_squared="+adjusted_r_squared.__str__())

print("mean_squared_error: "+mean_squared_error(y,y_pred).__str__())
print("r2_score: "+str(r2_score(y,y_pred)))

mpl.rcParams['legend.fontsize'] = 12
fig = plt.figure()
ax = fig.gca(projection ='3d')
ax.scatter(x[:, 0], x[:, 1], y, label ='y', s = 5,color ="dodgerblue")
ax.scatter(x[:, 0], x[:, 1], y_pred, label ='y_pred', s = 5,color ="orange")
ax.legend()
ax.view_init(45, 0)
plt.show()