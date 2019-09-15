import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.loadtxt("ex2x.dat")
y=np.loadtxt("ex2y.dat")

numberOfPoint=len(x)

sumOfxy, sumOfx2, sumOfx, sumOfy= 0, 0, 0, 0
for i in range(len(x)):
    sumOfxy+=(x[i] * y[i])
    sumOfx2+=(x[i] ** 2)
    sumOfx+=x[i]
    sumOfy+=y[i]
m= (sumOfxy - (sumOfx * sumOfy / numberOfPoint)) / (sumOfx2 - (sumOfx ** 2 / numberOfPoint))
c= (sumOfy / numberOfPoint) - ((m * sumOfx) / numberOfPoint)

plt.plot(x, y, 'o', label='Original data', markersize=10)
def my_generator():
    for var in x:
        yield var*m+c
yar= np.array(list(my_generator()))
plt.plot(x, yar, 'r', label='OSL')

#########################################
regressor = LinearRegression()
regressor.fit(x.reshape(-1, 1), y)

print("m="+str(m))
print("c="+str(c))
print("Lm="+str(regressor.coef_))
print("Lc="+str(regressor.intercept_))

y_pred = regressor.predict(x.reshape(-1, 1))

plt.plot(x, y_pred, 'b', label='Logistic Regression')

plt.legend()
plt.show()

