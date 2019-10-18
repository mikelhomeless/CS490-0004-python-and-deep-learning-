import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import StandardScaler

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset = pd.read_csv("diabetes.csv", header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, activation='relu')) # added two more hidden layers
my_first_nn.add(Dense(10, activation='relu'))
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))

# Original Score: [0.6597399512926737, 0.6770833333333334]
# New Score: [0.6286944051583608, 0.703125]
# After adding a couple more layers, the accuracy of my model went up, Though this isn't always bound to happen

# Question 2
dataset = pd.read_csv("breastcancer.csv")
dataset['diagnosis'] = dataset["diagnosis"].map({'M': 1, 'B': 0}).astype(int)
dataset = dataset.values[:, 1:32]

x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=0.25, random_state=87)

my_second_nn = Sequential()
my_second_nn.add(Dense(20, input_dim=30, activation='relu'))
my_second_nn.add(Dense(15, activation='relu'))
my_second_nn.add(Dense(10, activation='relu'))
my_second_nn.add(Dense(1, activation='sigmoid'))
my_second_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_second_nn_fitted = my_second_nn.fit(x_train, y_train, epochs=100, initial_epoch=0)

print(my_second_nn.summary())
print(my_second_nn.evaluate(x_test, y_test))
# Score: [0.3163792503463638, 0.8671328675496829]
# This is the raw score of the dataset without the data being scaled

# Question 3 Scaled data
scaler = StandardScaler()
dataset = dataset = pd.read_csv("breastcancer.csv")
dataset['diagnosis'] = dataset["diagnosis"].map({'M': 1, 'B': 0}).astype(int)
dataset = dataset.values[:, 1:32]

x_train, x_test, y_train, y_test = train_test_split(scaler.fit_transform(dataset[:, 1:]), dataset[:, 0], test_size=0.25, random_state=87)

my_second_nn = Sequential()
my_second_nn.add(Dense(20, input_dim=30, activation='relu'))
my_second_nn.add(Dense(15, activation='relu'))
my_second_nn.add(Dense(10, activation='relu'))
my_second_nn.add(Dense(1, activation='sigmoid'))
my_second_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_second_nn_fitted = my_second_nn.fit(x_train, y_train, epochs=100, initial_epoch=0)
print(my_second_nn.summary())
print(my_second_nn.evaluate(x_test, y_test))
# score: [0.3943201649439085, 0.9650349654517807]
# After scaling the data, it appears that the accuracy of the model has gone up