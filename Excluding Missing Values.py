import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Load data
data = pd.read_csv('webshop.csv')

#Create dummy variables for categorical variables
data = pd.get_dummies(data, columns=['Find_website', 'Device'])

#Fill missing values with column mean
data = data.fillna(data.mean())

#Fit regression model
X = data.drop('Purchase_Amount', axis=1)
y = data['Purchase_Amount']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

influence = model.get_influence()
cutoff = 4 / len(data)
cooks_d = influence.cooks_distance[0]
outliers = []

for i, d in enumerate(cooks_d):
    if d > cutoff:
        outliers.append(i)

data = data.drop(data.index[outliers]) # drop outliers

#Update X and y
X = data.drop('Purchase_Amount', axis=1)
y = data['Purchase_Amount']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

#Calculate VIF values for each independent variable
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#Display results
print(vif)
print(data.shape)

#Remove highly correlated variables
X = X.drop(['Pictures', 'Find_website_Other', 'Device_Mobile'], axis=1)
model = sm.OLS(y, X).fit()

#Checking the relationship between 'Time_Spent_on_Website' and 'Purchase_Amount'
plt.scatter(data['Time_Spent_on_Website'], data['Purchase_Amount'])
plt.xlabel('Time_Spent_on_Website')
plt.ylabel('Purchase Amount')
plt.show()

#logarithmic transformation to the 'Time_Spent_on_Website' variable to make the relationship linear
data['Time_Spent_on_Website'] = np.log(data['Time_Spent_on_Website'].replace(0, 1e-6))

plt.scatter(data['Time_Spent_on_Website'], data['Purchase_Amount'])
plt.xlabel('Log Time_Spent_on_Website')
plt.ylabel('Purchase Amount')
plt.show()

#APA Table
print(model.summary().tables[1])
