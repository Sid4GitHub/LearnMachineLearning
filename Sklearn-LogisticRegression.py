import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

x = np.array([0.09, 0.12, 0.17, 0.23, 0.31, 0.32, 0.33, 0.34, 0.38, 0.42, 0.43,
0.5, 0.51, 0.52, 0.55, 0.74, 0.78, 0.79, 0.8, 0.83, 0.84, 0.93, 0.95,
1.05, 1.05, 1.09, 1.11, 1.13, 1.2, 1.21, 1.24, 1.27, 1.33, 1.35, 1.38,
1.43, 1.45, 1.45, 1.48, 1.6, 1.62, 1.7, 1.76, 1.79, 1.79, 1.79, 1.94, 2])

y=np.array([7,7,7,7,7,7,7,7,7,7,7,7,7,7,
7,7,7,7,7,
7,7,7,7,7,26,26,26,26,
26,26,26,26,26,26,26,26,26,26,26,
26,26,26,26,26,26,26,26,26])

regressor = LogisticRegression()
regressor.fit(x.reshape(-1, 1), y)
m=regressor.coef_
c=regressor.intercept_

y_pred = regressor.predict(x.reshape(-1, 1))

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, y_pred, 'v', label='Original data', markersize=10)
plt.plot(x, y_pred, 'r', label='Fitted line')

plt.show()