from keras.layers import Embedding, Flatten
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values


#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
input_dim = X_train.shape[1]

# Number of features
print(input_dim)
model = Sequential()
model.add(layers.Dense(300,input_dim=input_dim, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

#  Changes made to get to work
# defined input_dim to be the vocabe size, with is also the second dimension of the training data, this way we have the correct number of inputs
# changed output layer activation function to be softmax for better classifications. Sigmoid isn't generally a great output activation function

# Question 2
# did not see increase in accuracy, but this is because I had to run with nerfed values for the program to run in reasonable time on my machine
max_review_length = max([len(s.split()) for s in df['review'].values])
vocab_size = len(tokenizer.word_index) + 1
padded_docs = pad_sequences(sentences, maxlen=max_review_length)
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_length))
model.add(Flatten())
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

# Question 3
newsgroups_train =fetch_20newsgroups(subset='train', shuffle=True)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(newsgroups_train.data)
sentences = tokenizer.texts_to_matrix(newsgroups_train.data)
y = newsgroups_train.target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

max_review_length = max([len(s.split()) for s in newsgroups_train.data])
vocab_size = len(tokenizer.word_index) + 1
padded_docs = pad_sequences(sentences, maxlen=max_review_length)
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_length))
model.add(Flatten())
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

