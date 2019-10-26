#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:28:06 2019

@author: archiefox
"""

#Importing modules: Numpy is used for several functions, Pandas for dataframes and matplotlib.pyplot for graphing
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import metrics, neighbors, linear_model, model_selection
from sklearn.model_selection import train_test_split
from patsy import dmatrices
import matplotlib.pyplot as plt

#Reading the data from Olympics, country regions, country GDPs and country populations
df = pd.read_csv('Output.csv')
df = df.dropna()

dfs = dict(tuple(df.groupby('Sport')))

weightlifters = df[df['Sport']=='Weightlifting']
non_weightlifters = df[df['Sport'] != 0]

weightlifters_weights = weightlifters.values[:, 5]
weightlifters_heights = weightlifters.values[:, 6]
non_weightlifters_weights = non_weightlifters.values[:, 5]
non_weightlifters_heights = non_weightlifters.values[:, 6]

plt.scatter(weightlifters_weights, weightlifters_heights, color='#e9a3c9', marker = 'o', label = 'Weightlifter', alpha=1)
plt.scatter(non_weightlifters_weights, non_weightlifters_heights, color='#91bfdb', marker = 'x', label = 'Non-weightlifter', alpha=0.2)
plt.legend(loc = 'lower right')
plt.show()
#for event in dfs:
    #print(event)
    #medal_winners = dfs[sport]
    #print(medal_winners[medal_winners['Medal_Won']==1].head().to_string())


'''
data1 = df.loc[df['Sex'] == "M", ['Sport', 'Age','Weight', 'Height']].drop_duplicates()
data1['target'] = np.where(data1['Sport'] == 'Weightlifting', 1, 0)

Y, X = dmatrices('target ~ 0 + Weight + Height', data = data1, return_type = 'dataframe')
y = Y['target'].values
accuracies = []
print(Y,X)
kfold = model_selection.StratifiedKFold(n_splits = 4, shuffle = True).split(X, y)

model = neighbors.KNeighborsClassifier(n_neighbors = 2,
                                      p = 2,
                                      weights = 'uniform')

for train, holdout in kfold:
    model.fit(X.iloc[train], y[train])
    prediction_on_test = model.predict(X.iloc[holdout])
    accuracies.append(metrics.accuracy_score(y[holdout], prediction_on_test))

print(np.mean(accuracies))'''

athletics = df.loc[df['Sex'] == 'M', ['Sport', 'Age', 'Weight', 'Height']].drop_duplicates()
features = ['Age', 'Height', 'Weight']
athletics['target'] = np.where(athletics['Sport'] == 'Weightlifting', 1, 0)
y = athletics['target']
X = athletics[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

neigh = neighbors.KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
y_pred = neigh.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("RMSE:", sqrt(mse))
print("R-squared score:", metrics.r2_score(y_test, y_pred))

'''
clf = linear_model.LogisticRegression(solver='liblinear').fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

for i in range(0,260):
    last_sample = X_test.iloc[i]
    for column, value in zip(features, last_sample):
        print(column + ': ' + str(value))
        last_sample_proba = y_pred_proba[i]
        print('Probability of 0:', last_sample_proba[0])
        print('Probability of 1:', last_sample_proba[1])
        print('Actual class:', int(y_test.iloc[i]))
general trends in weight/height
run loops for test size, n_folds etc'''