import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('winequality-red.csv')

#handling missing value by filling in with an average value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

##Build a linear model
y = np.log(train.quality)
X = data.drop(['quality'], axis=1)
Z = data.drop(['alcohol'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)


from sklearn import linear_model
lr = linear_model.LinearRegression()


model = lr.fit(X_train, y_train)

##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['quality'].sort_values(ascending=False)[:], '\n')

##Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# pivot and view all features
for feature in train.columns[:-1]:
    pivot = train.pivot_table(index=feature, values='quality', aggfunc=np.median)
    pivot.plot(kind='bar', color='blue')
    plt.show()

# collect only the features which are positively corrolated
revised_dataset = train[['alcohol', 'sulphates', 'citric acid', 'fixed acidity', 'residual sugar', 'quality']]
y = np.log(revised_dataset.quality)
x = revised_dataset.drop(['quality'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.33)
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
print ("R^2 is: \n", lr.score(x_test, y_test))
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
# It appears that the model did not improve, even with EDA