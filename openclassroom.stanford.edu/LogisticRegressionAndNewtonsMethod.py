import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#import pandas as pd

x1x2=np.loadtxt("ex4x.dat")
y=np.loadtxt("ex4y.dat")
x1=x1x2[:,0]
x2=x1x2[:,1]

plt.figure("Original")
plt.scatter(x1,x2, c=y, s=20, cmap='viridis')

regressor = LogisticRegression()
regressor.fit(x1x2,y)
y_pred = regressor.predict(x1x2)

min_x1=int(np.min(x1))
max_x1=int(np.max(x1))

min_x2=int(np.min(x2))
max_x2=int(np.max(x2))

def getX1Pints():
    for i in range(min_x1,max_x1):
        yield i
contigiousX1Points=np.array(list(getX1Pints()))
def getX2Pints():
    for i in range(min_x2,max_x2):
        yield i
contigiousX2Points=np.array(list(getX2Pints()))

def getAllxy():
    for i in contigiousX1Points:
        for j in contigiousX2Points:
            yield (i,j)

xy=np.array(list(getAllxy()))

re=regressor.predict_proba(xy)[:,0]

posXY=[]
for i in range(len(xy)):
    if(re[i]>=.5):
        posXY.append(xy[i])

xx=list(set(np.array(posXY)[:,0]))

xy=[]

for i in xx:
    maxY=0
    for j in posXY:
        if(j[0]==i):
            if (j[1])>maxY:
                maxY=j[1]
    print("i=" + str(i) + "  maxY=" + str(maxY))
    if(maxY<85):
        xy.append((i,maxY))
print("min_x1="+str(min_x1)+" :min_x2="+str(min_x2)+"  :max_x1="+str(max_x1)+"  :max_x2"+str(max_x2))
xy=np.array(xy)
'''
df=pd.DataFrame(data=posXY[0:,0:],
            index=[i for i in range(posXY.shape[0])],
            columns=['f'+str(i) for i in range(posXY.shape[1])])
print(df.head())
print("---------------")
xx=df.groupby("f0")['f1'].idxmax().values

print(df.filter(id=xx))
'''
#y_predForDecisionBoundary = regressor.predict(contigiousX1Points)


plt.figure("LogisticRegression")
plt.scatter(x1,x2, c=y_pred, s=20, cmap='cool')
plt.plot(xy[:,0], xy[:,1], 'r', label='Fitted line')
plt.show()