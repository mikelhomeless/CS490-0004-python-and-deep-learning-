import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

titanic_train_data = pd.read_csv('train_preprocessed.csv')
titanic_test_data = pd.read_csv('test_preprocessed.csv')

# Question 1
print(titanic_train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Evaluates to:
#    Sex  Survived
# 1    1  0.742038
# 0    0  0.188908

# with the out put given by the above function, it is clear that there is a high correlation between survival and sex
# Therefore, we should keep sex as a feature for our model


data_frame = pd.read_csv('glass.csv')
x_train, x_test = train_test_split(data_frame, test_size=0.2)
y_train = x_train['Type']
x_train = x_train.drop('Type', axis=1)

y_test = x_test['Type']
x_test = x_test.drop('Type', axis=1)

# Question 2
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
accuracy = round(gnb.score(x_train, y_train) * 100, 2)
print("GNB accuracy is:", accuracy)
print("GNB classification report:\n", classification_report(y_test, y_pred))

# Question 3
svc = SVC()
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
accuracy = round(svc.score(x_train, y_train) * 100, 2)
print("svm accuracy is:", accuracy)
print("SVM classification report:\n", classification_report(y_test, y_pred))

# SVM has a clear advantage over the naive bayes algorithm in the overall accuracy. The naive bayes algorithm works best under the condition
# that the features of a class are independent. That being the case, the advantage SVM has over the naive bayes algorithm
# in this example is most likely due to features being dependent.