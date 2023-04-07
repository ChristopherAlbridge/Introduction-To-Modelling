import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Load the data from the provided URL
data = pd.read_csv('webshop.csv')

#Impute missing values in the selected columns
cols_to_impute = ['Time_Spent_on_Website', 'Number_of_products_browsed', 'Pictures', 'Shipping_Time', 'Review_rating', 'Ease_of_purchase', 'Age']
imputer = SimpleImputer(strategy='mean')
data[cols_to_impute] = imputer.fit_transform(data[cols_to_impute])

#Standardize the selected columns
cols_to_scale = ['Purchase_Amount', 'Time_Spent_on_Website', 'Pictures', 'Shipping_Time', 'Review_rating', 'Ease_of_purchase', 'Age']
scaler = StandardScaler()
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

#Create dummy variables for the 'Find_website' and 'Device' columns
dummies = pd.get_dummies(data[['Find_website', 'Device']])
data = pd.concat([data, dummies], axis=1)

#Checking the Multicollinearity by getting the design matrix of independent variables
X = data[['Time_Spent_on_Website', 'Number_of_products_browsed', 'Pictures', 'Shipping_Time', 'Review_rating', 'Ease_of_purchase', 'Age']]

# Calculate VIF for each variable
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Variable"] = X.columns

# Print the VIF values
print(vif)

#Fitting regression models
modela = sm.ols('Purchase_Amount ~ Time_Spent_on_Website', data=data).fit()
modelb = sm.ols('Purchase_Amount ~ Number_of_products_browsed', data=data).fit()
modelc = sm.ols('Purchase_Amount ~ Pictures', data=data).fit()
modeld = sm.ols('Purchase_Amount ~ Shipping_Time', data=data).fit()
modele = sm.ols('Purchase_Amount ~ Review_rating', data=data).fit()
modelf = sm.ols('Purchase_Amount ~ Find_website', data=data).fit()
modelg = sm.ols('Purchase_Amount ~ Ease_of_purchase', data=data).fit()
modelh = sm.ols('Purchase_Amount ~ Age', data=data).fit()
modeli = sm.ols('Purchase_Amount ~ Device', data=data).fit()

#Multiple regression
model1 = smf.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Find_website + Ease_of_purchase + Age + Device', data=data).fit()
print(model1.summary())

#Identifying and removing outliers using Cook's D
infl = model1.get_influence()
summ_df = infl.summary_frame()
data['Outlier'] = summ_df['cooks_d'] > 4/len(data)
data = data[data['Outlier'] == False]

#Fitting the regression model with outliers removed
model2 = smf.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Find_website + Ease_of_purchase + Age + Device', data=data).fit()
print(model2.summary())

#Non-linear regression
sns.regplot(x='Time_Spent_on_Website', y='Purchase_Amount', data=data, lowess=True)
sns.regplot(x='Number_of_products_browsed', y='Purchase_Amount', data=data, lowess=True)
sns.regplot(x='Pictures', y='Purchase_Amount', data=data, lowess=True)
sns.regplot(x='Shipping_Time', y='Purchase_Amount', data=data, lowess=True)
sns.regplot(x='Review_rating', y='Purchase_Amount', data=data, lowess=True)
sns.regplot(x='Ease_of_purchase', y='Purchase_Amount', data=data, lowess=True)
sns.regplot(x='Age', y='Purchase_Amount', data=data, lowess=True)

#The Correlation
correlations = data[['Purchase_Amount', 'Time_Spent_on_Website', 'Pictures', 'Shipping_Time', 'Review_rating', 'Ease_of_purchase', 'Age']].corr()
print(correlations)

#Fitting a multiple linear regression model
model = smf.ols('Purchase_Amount ~ Time_Spent_on_Website + Number_of_products_browsed + Pictures + Shipping_Time + Review_rating + Find_website + Ease_of_purchase + Age + Device', data=data).fit()
#Printing the summary statistics for the model
print(model.summary())

#Create a new row of data with the customer's characteristics
new_data = pd.DataFrame({
    'Purchase_Amount': [0],
    'Time_Spent_on_Website': [723],
    'Number_of_products_browsed': [20],
    'Pictures': [3.4],
    'Shipping_Time': [2.6],
    'Review_rating': [4.5],
    'Find_website_Friends or Family': [1],
    'Find_website_Search_Engine': [0],
    'Find_website_Social_Media_Advertisement': [0],
    'Find_website_Other': [0],
    'Ease_of_purchase': [4],
    'Age': [35],
    'Device_PC': [1]
})

# Standardize the columns using the same scaler object used for the training data
new_data[cols_to_scale] = scaler.transform(new_data[cols_to_scale])

# Make the prediction using the fitted regression model
prediction = model.predict(new_data)
# The predicted purchase amount is in standardized units, so we need to inverse transform it to get the actual amount in dollars
predicted_purchase_amount = scaler.inverse_transform(np.array(prediction))[0]

print(f'The predicted purchase amount for the new customer is ${predicted_purchase_amount:.2f}.')


#APA TABLE


#Creating a list of the models to include in the table
model.summary()
