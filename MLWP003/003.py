import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC

data = pd.read_csv('wineQT.csv')

# Making sure that all the data has the right datatype

print("Data Types of each of our data's columns: ")
for value in data.keys():
    print(data[str(value)].dtype)
          
# Making sure there are no duplicates  
 
print("Number of trues and falses indicating duplications") 
print((data.duplicated()).value_counts())


# So let's try to predict the quality based on other 
# features of the wine

# Classifying the data pool as data (X) and target (y)
# and making sure they are of the appropriate shape

X = data.drop(columns = 'quality')
y = data['quality']
print(X.shape)
print(y.shape)

# Dividing data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_test.shape)

# Creating the model

# Logistic regression (does very bad)
logreg = LogisticRegression(C=0.100).fit(X_train, y_train)
print(logreg.score(X_train, y_train))
print(logreg.score(X_test, y_test))

# Lasso regression (Does worse)
lasso = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))
print(np.sum(lasso.coef_ != 0))

# Linear svc model (Does the same as lasso model) 
# WTF AM I DOING WRONG
LSVC = LinearSVC(max_iter=1000).fit(X_train, y_train)
print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))




