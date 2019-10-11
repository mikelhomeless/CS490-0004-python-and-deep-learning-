from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('Bayes Score: ', score)

# Replacing with a KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train_tfidf, twenty_train.target)

predicted = classifier.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print('KNClassifier Score: ', score)

# changing tfidf vectorizer
tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

classifier = KNeighborsClassifier()
classifier.fit(X_train_tfidf, twenty_train.target)

predicted = classifier.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print('KNClassifier Score with tfidf vectorizer changed: ', score)

# Change stop words
tfidf_Vect = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

classifier = KNeighborsClassifier()
classifier.fit(X_train_tfidf, twenty_train.target)

predicted = classifier.predict(X_test_tfidf)
score = metrics.accuracy_score(twenty_test.target, predicted)
print('KNClassifier Score with new stop words: ', score)