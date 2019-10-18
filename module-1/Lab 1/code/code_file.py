# 1) soln:
temp = " "
dictionary = {}
for j in range(len(str)):
    for i in range(j, len(str)):
        if not (str[i] in temp):
            temp += str[i]
        else:
            dictionary[temp] = len(temp)
            temp = ''
            break

max_val = max(dictionary.values())
list1 = []
for key, val in dictionary.items():
    if max_val == val:
        list1.append((key, val))

# 2) soln:
# Create a dictionary with keys as names and values as list of (subjects, marks) in sorted order.

student_list = [
    ('John', ('Physics', 80)) ,
    ('Daniel', ('Science', 90)),
    ('John', ('Science', 95)),
    ('Mark', ('Maths', 100)),
    ('Daniel', ('History', 75)),
    ('Mark', ('Social', 95))
]

student_dict = dict()

for student in student_list:
    if student[0] not in student_dict:
        student_dict[student[0]] = list()

    student_dict[student[0]].append(student[1])

for student in student_dict:
    print(student, ":", sorted(student_dict[student]))

# 3)
# soln:
# > I have created the Library management system, which has 5 different
#   classes namely Person, Student, Librarian, Book, Borrow_book.
# > Person is the main class and the classes Student and Librarian
#   have inherited (single inheritance) the Person class.
# > Borrow_Book class implements multiple inheritance with base class
#   Student, Book.
# > Declared private data member StudentCount in student class to count
#   number of student objects created.
# > Used super call in class Librarian to initialize Person class object.
# > Used a private member __numBooks for keeping the track of books.


class Person:
    def __init__(self, name, email):
     self.name = name
     self.email = email

    def display(self):
        print("Name: ", self.name)
        print("Email: ", self.email)


#Inheritance concept where student, is inheriting the person class
class Student(Person):
    StudentCount = 0

    def __init__(self, name, email, student_id):
        Person.__init__(self, name, email)
        self.student_id = student_id
        Student.StudentCount += 1


#super class in the librarian class
class Librarian(Person):
    StudentCount = 0

    def __init__(self, name, email, employee_id):
        #super call where librarian class is inheriting the Person class
        super().__init__(name, email)
        self.employee_id = employee_id


class Book:
    #creating a private number
    num_books = 0 #private number

    def __init__(self, book_name, author, book_id):
        self.book_name = book_name
        self.author = author
        self.book_id = book_id
        Book.num_books +=1 #keeps track of which student or staff has book checked

    def display(self):
        print('Author: {}, Title: {}, ID {}'.format(self.author, self.book_name, self.book_id))


#multiple inheritance concept for Borrow_Book class
class Borrow_Book:
    def __init__(self, student, book):
        self.student = student
        self.book = book

    def display(self):
        print("Borrowed Book Details:")
        self.student.display()
        self.book.display()

#creating instance of all classes
students = []
books = []
employees = []
rentals = []

students.append(Student('Anisha', 'anishax01@gmail.com', 123))
employees.append(Librarian('Michael', 'michaely02@gmail.com', 456))
books.append(Book('Notebook', 'Syed', 789))

student = students[0]
rentals.append(Borrow_Book(student, books[0]))

print('Employees: ', end='\n\n')
employees[0].display()

print('Current rental: ', end='\n\n')
rentals[0].display()

# 4)
from bs4 import BeautifulSoup
import urllib.request
import os

url = "https://scikit-learn.org/stable/modules/clustering.html#clustering"
#makes a request to open the url and set the contents of the url to a variable
source_code = urllib.request.urlopen(url)
plain_text = source_code

#Using BeautifulSoup to parse the html page
soup = BeautifulSoup(plain_text, "html.parser")

print(soup.get_text().strip())

# 5
#   a)
# Soln:
# Dataset: Car Evaluation Data Set from UC Irvine Machine Learning Repository
#
# Dataset Link: http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
#
# Features:
#
# buying - vhigh, high, med, low
#
# maint - vhigh, high, med, low
#
# doors - 2, 3, 4, 5more
#
# persons - 2, 4, more
#
# lug_boot - small, med, big
#
# safety - low, med, high
#
#
# Target variable:
#
# class values - unacc, acc, good, vgood

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('use_inf_as_na', True)
dataset = pd.read_csv('car.csv', header=None)
dataset.columns = ["buying", "maint", "doors", "persons", "lugBoot", "safety", "classValue"]
# replace anything that is inf to nan
dataset.replace([np.inf, -np.inf], np.nan)
# remove all na
dataset = dataset.dropna()

# change non numerical data into numerical
dataset['buying'] = dataset['buying'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}).astype(int)
dataset['maint'] = dataset['maint'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}).astype(int)
dataset['doors'] = dataset['doors'].map({'2': 0, '3': 1, '4': 2, '5more': 3}).astype(int)
dataset['persons'] = dataset['persons'].map({'2': 0, '4': 1, 'more': 2}).astype(int)
dataset['lugBoot'] = dataset['lugBoot'].map({'small': 0, 'med': 1, 'big': 2}).astype(int)
dataset['safety'] = dataset['safety'].map({'low': 0, 'med': 1, 'high': 2}).astype(int)
dataset['classValue'] = dataset['classValue'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}).astype(int)


# identify correlation of features by doing pivots
for feature in dataset.columns[:-1]:
    yeet = dataset[[feature, 'classValue']].groupby(feature, as_index=False).mean().sort_values(by='classValue', ascending=False)
    print(yeet, end='\n\n\n\n')

# step through each of the features and visualize them against the end state
for feature in dataset.columns[:-1]:
    g = sns.FacetGrid(dataset, col='classValue')
    g.map(plt.hist, feature)
    plt.show()


from sklearn.model_selection import train_test_split
x_train = dataset.drop('classValue', axis=1)
y_train = dataset['classValue']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("GNB classification report:\n", classification_report(y_test, y_pred))

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("SVM classification report:\n", classification_report(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("KNN classification report:\n", classification_report(y_test, y_pred))

# it is clear that svm and knn both work better than bayes

# Question 6
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')



x = data.iloc[:,1:]
# y = dataset.iloc[:,-1]

#handling missing value by filling in with an average value

z = x.apply(lambda x: x.fillna(x.mean()),axis=0)

##building the model
from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(z)


wcss= []  ##Within Cluster Sum of Squares
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,
max_iter=300,random_state=0)
    kmeans.fit(z)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

print("Based on the Elbow method K=3 is the best")

#Silhouette score
# predict the cluster for each data point
y_cluster_kmeans = km.predict(z)
from sklearn import metrics
score = metrics.silhouette_score(z, y_cluster_kmeans)
print("The Silhouette score is",score)

from sklearn import preprocessing
scaler =preprocessing.StandardScaler()
scaler.fit(z)
X_scaled_array=scaler.transform(z)
X_scaled=pd.DataFrame(X_scaled_array, columns =z.columns)

print("Feature Scaling",X_scaled)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(z)#
#Apply transform to both the training set and the test set.
x_scaler= scaler.transform(z)

from sklearn.decomposition import PCA# Make an instance of the Model
pca= PCA(2)
X_pca= pca.fit_transform(x_scaler)
print("Applying PCA on the dataset:",X_pca)

# Question 7
file_name = input('Enter file path you want to open: ')

# the test file was converted to utf-8
text = ''
with open(file_name) as file:
    text = file.read()

# tokenize and apply lemmatization
import nltk
word_tokens = nltk.word_tokenize(text)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lem_words = [lemmatizer.lemmatize(word) for word in word_tokens]
# first 10
print(lem_words[:10])

# find all trigrams
from nltk.util import trigrams
trigrams = trigrams(lem_words)

from collections import defaultdict
trigram_dictionaries = defaultdict(int)
for x in trigrams:
    trigram_dictionaries[x] += 1

top_ten_trigrams = sorted(trigram_dictionaries.items(), key=lambda x: x[1], reverse=True)[:10]
print('Top ten trigrams')
print(top_ten_trigrams)

# extracting all the sentences in the file
sentence_tokens = nltk.sent_tokenize(text)


# gather all sentences that contain the top ten trigrams. Some sentences are repeated twice because they contain multiple of the common trigrams
sentences = [sentence for trigram in top_ten_trigrams for sentence in sentence_tokens if ' '.join(trigram[0]) in sentence]
print('sentences that contain the trigrams')
print(' '.join(sentences))

#question 8
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