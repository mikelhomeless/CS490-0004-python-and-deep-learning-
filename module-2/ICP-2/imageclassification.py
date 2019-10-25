# fix for an issue with running multiple instances of libiomp5.dylib on MacOS
import os
import platform
if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import everything else
from keras import Sequential
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

print(train_images.shape[1:])
#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
print(dimData)
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /= 255.0
test_data /= 255.0
#change the labels frominteger to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

# Question 1 Start plotting accuracy and loss of both training and validation data
metrics = history.history
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.plot(metrics['val_acc'], label="Validation Accuracy")
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['acc'], label="Training Accuracy")
plt.legend()
plt.show(block=True)

# Question 2 Plot image
y1 = model.predict_classes(test_data[[0], :])
plt.imshow(test_images[0, :], cmap='gray')
plt.title('Ground Truth: {} Predicted Value: {}'.format(test_labels[0], y1))
plt.show(block=True)

print("Predicted value: {}. True value {}".format(y1, test_labels[0]))

# Question 3 Change hidden layer activation functions
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(dimData,)))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
new_history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

new_metrics = new_history.history
print("new accuracy: {}, old_accuracy: {}, new_loss: {}, old_loss: {}".format(new_metrics['acc'][-1], metrics['acc'][-1], new_metrics['loss'][-1], metrics['loss'][-1]))
