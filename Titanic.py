# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 00:04:58 2018

@author: Nando's Lenovo
"""
#Importing Libraries
import numpy as np
import pandas as pd

#Importing Data
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
#Data Preprocessing
dataset = pd.concat([dataset_train, dataset_test], sort = False)
#Drroping Unwanted Data
dataset = dataset.drop(['Ticket', 'PassengerId','Cabin'], axis = 1 )
#Handling NaN Values
dataset[['Age']] = dataset[['Age']].fillna(value = dataset[['Age']].median()) 
dataset[['Fare']] = dataset[['Fare']].fillna(value = dataset[['Fare']].median())
dataset[['Embarked']] = dataset[['Embarked']].fillna(value=dataset['Embarked'].value_counts().idxmax())
#Handling Names
dataset['Title'] = dataset.Name.map(lambda x: x.split(',')[1].split( '.' )[0].strip())
#dataset['Title'].value_counts()
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
dataset.Title.loc[ (dataset.Title !=  'Master') & (dataset.Title !=  'Mr') & (dataset.Title !=  'Miss') 
             & (dataset.Title !=  'Mrs')] = 'Others'
dataset = pd.concat([dataset, pd.get_dummies(dataset['Title'])], axis=1).drop(labels=['Name'], axis=1)
#dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
dataset = dataset.drop(['Title','Master','Miss','Mr','Mrs','Others'], axis = 1 )
#X = dataset.iloc[:, 2:9].values
#Y = dataset.iloc[:,1:2].values
#Categorical encoder
from sklearn.preprocessing import LabelEncoder
labelencoder_sex = LabelEncoder()
dataset['Sex'] = labelencoder_sex.fit_transform(dataset['Sex']) #Male = 1, Female = 0
labelencoder_e = LabelEncoder()
dataset['Embarked'] = labelencoder_e.fit_transform(dataset['Embarked'])
#labelencoder_t = LabelEncoder()
#dataset['Title'] = labelencoder_t.fit_transform(dataset['Title'])  
#Converting numeric values of Embarked to Binary Classifications
onehotencoder = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
dataset = dataset.drop('Embarked', axis=1)
dataset = pd.concat([dataset, onehotencoder], axis = 1, sort = False)
dataset = dataset.drop(['Embarked_0'], axis = 1 )
#Converting pd to np
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:,0:1].values

X_train = X[:891, 1:]
X_test  = X[891:, 1:]
y_train = y[:891, 0:1]
y_test  = y[891:, 0:1]

#Spliting the dataset into Training and Testing
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.319, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Building ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def my_model(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'uniform', input_dim = 7 ))
    #classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'uniform'))
    #classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'uniform'))
    #classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = my_model)
parameters = {'batch_size':[25, 32], 'epochs':[100, 500], 'optimizer': ['adam', 'rmsprop'] }
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10 )
grid_search_result = grid_search.fit(X_train, y_train)
best_parameters = grid_search_result.best_params_
best_score = grid_search_result.best_score_

#Prediction after Tuning the ANN
classifier = Sequential()
classifier.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'uniform', input_dim = 7))
#classifier.add(Dropout(0.1))
classifier.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'uniform'))
#classifier.add(Dropout(0.1))
classifier.add(Dense(units = 4, activation = 'relu', kernel_initializer = 'uniform'))
#classifier.add(Dropout(0.1))
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train,y_train, batch_size = 32, epochs = 100)
#Evaluate the model
score = classifier.evaluate(X_train, y_train)
print('test loss', score[0])
print('test accuracy', score[1])
#Prediction on Test set
y_pred = classifier.predict(X_test)   
y_pred = (y_pred > 0.4)







