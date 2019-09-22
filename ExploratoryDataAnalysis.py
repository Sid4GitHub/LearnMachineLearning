import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
filePath="./data-set/kaggle/house-prices-advanced-regression-techniques/train.csv"
df = pd.read_csv(filePath)
print(df.head(1))
#get numarical columns
print("Numarical columns:")
print(list(df._get_numeric_data().columns))
#get non-numeric data
print("Non-Numarical columns:")
print(list(set(df.columns)-set(df._get_numeric_data().columns)))


df.fillna(df.mean(), inplace=True)
#kde: kernel desnsity estimation [to put a continious function]
for i in [str(j) for j in set(df._get_numeric_data())]:
  sns.distplot(df[i],kde=True)
  plt.figure()

#Boxplot
for i in [str(j) for j in set(df._get_numeric_data())]:
  sns.boxplot(df[i])
  plt.figure()

plt.figure(figsize=(16,8))
sns.countplot(y=df['Neighborhood'])

#Multivariate
#numeric vs numeric
sns.lmplot('GrLivArea','SalePrice',data=df,fit_reg=True)
plt.figure()
sns.jointplot('TotalBsmtSF','SalePrice',data=df,kind='reg')

plt.figure()
sns.jointplot('YearBuilt','SalePrice',data=df,kind='hex')

#Multivariate
#numeric vs numeric
sns.lmplot('GrLivArea','SalePrice',data=df,fit_reg=True)
plt.figure()
sns.jointplot('TotalBsmtSF','SalePrice',data=df,kind='reg')

plt.figure()
sns.jointplot('YearBuilt','SalePrice',data=df,kind='hex')

#removal of corelated variablle
k=10
cols=df.corr().nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm=df[cols].corr()
plt.figure(figsize=[20,12])
sns.heatmap(data=cm,cmap='viridis',annot=True)

plt.figure()
sns.pairplot(df[cols])


#numeric vs categorical
#boxplot
plt.figure(figsize=(15,8))
plt.xticks(rotation=60)
sns.boxplot('Neighborhood','SalePrice',data=df)

plt.figure(figsize=(15,8))
sns.swarmplot('OverallQual','SalePrice',data=df)

#Categorical vs categorical
crosstab=pd.crosstab(index=df['Neighborhood'],columns=df['OverallQual'])
print(crosstab)

#stacked-bar plot
crosstab.plot(kind='bar',figsize=(15,8),stacked=True,colormap='Paired')


plt.show()
