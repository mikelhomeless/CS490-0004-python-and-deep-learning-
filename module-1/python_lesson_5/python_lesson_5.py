import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Question 1
plt.style.use(style='ggplot')

train = pd.read_csv('train.csv')
x = train['GarageArea']
y = train['SalePrice']
plt.scatter(x, y)

# passing in block=False so that gui doesn't block the execution of the rest of the script
plt.show(block=False)
plt.pause(3)
plt.close()

# as we saw in the graph, outliers are areas greater than 1,200
# getting rid of all garage areas greater than 1,200
print('Removed column indecies:', train[train.GarageArea > 1200].index.to_list())
train = train[train.GarageArea < 1_200]

# Question 2
train = pd.read_csv('winequality-red.csv')
train = train.select_dtypes(include=[np.number]).dropna()
correlations = train.corr()
print('Top 3 corrolations are:\n', correlations['quality'].sort_values(ascending=False)[1:4])

y = train.quality
x = train.drop('quality', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

regression = linear_model.LinearRegression()
model = regression.fit(x_train, y_train)
print('The R2 score is:', model.score(x_test, y_test))
pred = model.predict(x_test)
print('The mean-squared-error is:', mean_squared_error(y_test, pred))
