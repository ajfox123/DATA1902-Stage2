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
from sklearn import metrics, linear_model, model_selection, neighbors
from sklearn.model_selection import train_test_split, GridSearchCV
from patsy import dmatrices
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Spectral4

df = pd.read_csv('Output.csv')
df = df.dropna()

medal_mappings = {'No Medal':0, 'Bronze':1, 'Silver':3, 'Gold':5}
df['Weighted_Medal'] = df['Medal'].map(medal_mappings)

team_events = pd.pivot_table(df,
                             index = ['Team', 'Year', 'Event'],
                             columns = 'Medal',
                             values = 'Medal_Won',
                             aggfunc = 'sum',
                             fill_value = 0).drop('No Medal', axis = 1).reset_index()

team_events = team_events.loc[team_events['Gold'] > 1, :]
team_event = team_events['Event'].unique()
remove_events = ["Gymnastics Women's Balance Beam", "Gymnastics Men's Horizontal Bar", 
                 "Swimming Women's 100 metres Freestyle", "Swimming Men's 50 metres Freestyle"]
team_events = list(set(team_events) - set(remove_events))
team_event_mask = df['Event'].map(lambda x: x in team_events)
single_event_mask = [not i for i in team_event_mask]
medal_won_mask = df['Medal_Won'] == 1
df['Team_Event'] = np.where(team_event_mask & medal_won_mask, 1, 0)
df['Single_Event'] = np.where(single_event_mask & medal_won_mask, 1, 0)
df['Event_Category'] = df['Single_Event'] + df['Team_Event']

medal_tally_agnostic = df.groupby(['Year', 'Team', 'Event', 'Medal'])[['Medal_Won', 'Event_Category']].agg('sum').reset_index()
medal_tally_agnostic['Medal_Won_Corrected'] = medal_tally_agnostic['Medal_Won']/medal_tally_agnostic['Event_Category']
medal_tally = medal_tally_agnostic.groupby(['Year', 'Team'])['Medal_Won_Corrected'].agg('sum').reset_index()

top_countries = medal_tally.groupby('Team')['Medal_Won_Corrected'].agg('sum').reset_index().sort_values('Medal_Won_Corrected', ascending = False).head(n=4)['Team']
top_countries = ['USA', 'Russia', 'Germany', 'China']

top_countries_mask = df['Team'].map(lambda x: x in top_countries)

year_team_athelete = df.loc[top_countries_mask, ['Year', 'Team', 'Name']].drop_duplicates()
contingent_size = pd.pivot_table(year_team_athelete,
                                 index = 'Year',
                                 columns = 'Team',
                                 values = 'Name',
                                 aggfunc = 'count')
year_team_medals = pd.pivot_table(medal_tally,
                                  index = 'Year',
                                  columns = 'Team',
                                  values = 'Medal_Won_Corrected',
                                  aggfunc = 'sum')[top_countries]





#Plotting contingent size vs medal tally for top countries
for country in top_countries:
    plt.figure(figsize=(10,10))
    contingent_size[country].plot(linestyle = '-', marker = 'o', linewidth = 2, color = 'red', label = 'Contingent Size')
    year_team_medals[country].plot(linestyle = '-', marker = 'o', linewidth = 2, color = 'black', label = 'Medal Tally')
    plt.xlabel('Olympic Year')
    plt.ylabel('Number of Athletes/Medal Tally')
    plt.title('Team '+country+'\nContingent Size vs Medal Tally')
    plt.legend(loc = 'best')
    plt.show()





#Correlation between contingent size and medal tally for all countries
all_countries = df['Team'].unique()
all_countries_mask = df['Team'].map(lambda x: x in all_countries)
year_team_athelete = df.loc[all_countries_mask, ['Year', 'Team', 'Name']].drop_duplicates()
contingent_size = pd.pivot_table(year_team_athelete,
                                 index = 'Year',
                                 columns = 'Team',
                                 values = 'Name',
                                 aggfunc = 'count')
year_team_medals_unstack = year_team_medals.unstack().reset_index()
year_team_medals_unstack.columns = ['Team','Year', 'Medal_Count']
contingent_size_unstack = contingent_size.unstack().reset_index()
contingent_size_unstack.columns = ['Team','Year', 'Contingent']
contingent_medals = contingent_size_unstack.merge(year_team_medals_unstack,
                                                 left_on = ['Team', 'Year'],
                                                 right_on = ['Team', 'Year'])
print(contingent_medals[['Contingent', 'Medal_Count']].corr())





stats = {'largest_avg_height':['',0],
         'lowest_avg_height':['',0],
         'largest_avg_weight':['',0],
         'lowest_avg_weight':['',0],
         'greatest_std_height':['',0],
         'lowest_std_height':['',0],
         'largest_std_weight':['',0],
         'lowest_std_weight':['',0]}

sports = dict(tuple(df.groupby('Sport')))
for sport in sports:
    avg_height = np.mean(sports[sport]['Height'])
    avg_weight = np.mean(sports[sport]['Weight'])
    std_height = np.std(sports[sport]['Height'])
    std_weight = np.std(sports[sport]['Weight'])
    if avg_height > stats['largest_avg_height']:
        stats['largest_avg_height'] = [sport, avg_height]
    if avg_height < stats['lowest_avg_height']:
        stats['lowest_avg_height'] = [sport, avg_height]
    if avg_weight > stats['largest_avg_weight']:
        stats['largest_avg_weight'] = [sport, avg_weight]
    if avg_weight < stats['lowest_avg_weight']:
        stats['lowest_avg_weight'] = [sport, avg_weight]
    if std_height > stats['largest_std_height']:
        stats['largest_std_height'] = [sport, std_height]
    if std_height < stats['lowest_std_height']:
        stats['lowest_avg_height'] = [sport, avg_weight]
    if std_height > stats['largest_std_height']:
        stats['largest_avg_height'] = [sport, avg_height]
    if std_weight < stats['lowest_std_height']:
        stats['lowest_avg_height'] = [sport, avg_weight]













































'''
team_events = pd.pivot_table(df,
                             index = ['Team', 'Year', 'Event'],
                             columns = 'Medal',
                             values = 'Medal_Won',
                             aggfunc = 'sum',
                             fill_value = 0).drop('No Medal', axis = 1).reset_index()

team_events = team_events.loc[team_events['Gold'] > 1, :]
team_event = team_events['Event'].unique()
remove_events = ["Gymnastics Women's Balance Beam", "Gymnastics Men's Horizontal Bar", 
                 "Swimming Women's 100 metres Freestyle", "Swimming Men's 50 metres Freestyle"]
team_events = list(set(team_events) - set(remove_events))
team_event_mask = df['Event'].map(lambda x: x in team_events)
single_event_mask = [not i for i in team_event_mask]
medal_won_mask = df['Medal_Won'] == 1
df['Team_Event'] = np.where(team_event_mask & medal_won_mask, 1, 0)
df['Single_Event'] = np.where(single_event_mask & medal_won_mask, 1, 0)
df['Event_Category'] = df['Single_Event'] + df['Team_Event']

medal_tally_agnostic = df.groupby(['Year', 'Team', 'Event', 'Medal'])[['Medal_Won', 'Event_Category']].agg('sum').reset_index()
medal_tally_agnostic['Medal_Won_Corrected'] = medal_tally_agnostic['Medal_Won']/medal_tally_agnostic['Event_Category']
medal_tally = medal_tally_agnostic.groupby(['Year', 'Team'])['Medal_Won_Corrected'].agg('sum').reset_index()

top_countries = ['USA', 'China', 'Russia', 'Germany', 'Australia']

countries = df['Team'].unique()

top_countries_mask = df['Team'].map(lambda x: x in top_countries)

year_team_athelete = df.loc[top_countries_mask, ['Year', 'Team', 'Name']].drop_duplicates()
contingent_size = pd.pivot_table(year_team_athelete,
                                 index = 'Year',
                                 columns = 'Team',
                                 values = 'Name',
                                 aggfunc = 'count')
year_team_medals = pd.pivot_table(medal_tally,
                                  index = 'Year',
                                  columns = 'Team',
                                  values = 'Medal_Won_Corrected',
                                  aggfunc = 'sum')[countries]



#Plotting contingent size vs medal tally for top countries
for country in top_countries:
    contingent_size[country].plot(linestyle = '-', marker = 'o', linewidth = 2, color = 'red', label = 'Contingent Size')
    year_team_medals[country].plot(linestyle = '-', marker = 'o', linewidth = 2, color = 'black', label = 'Medal Tally')
    plt.xlabel('Olympic Year')
    plt.ylabel('Number of Athletes/Medal Tally')
    plt.title('Team '+country+'\nContingent Size vs Medal Tally')
    plt.legend(loc = 'best')
    plt.show()



#Correlation between contingent size and medal tally for all countries
all_countries_mask = df['Team'].map(lambda x: x in countries)
year_team_athelete = df.loc[all_countries_mask, ['Year', 'Team', 'Name']].drop_duplicates()
contingent_size = pd.pivot_table(year_team_athelete,
                                 index = 'Year',
                                 columns = 'Team',
                                 values = 'Name',
                                 aggfunc = 'count')
year_team_medals_unstack = year_team_medals.unstack().reset_index()
year_team_medals_unstack.columns = ['Team','Year', 'Medal_Count']
contingent_size_unstack = contingent_size.unstack().reset_index()
contingent_size_unstack.columns = ['Team','Year', 'Contingent']
contingent_medals = contingent_size_unstack.merge(year_team_medals_unstack,
                                                 left_on = ['Team', 'Year'],
                                                 right_on = ['Team', 'Year'])
print(contingent_medals[['Contingent', 'Medal_Count']].corr())




#Linear Regression Model for an athlete winning a medal in Basketball
sports = dict(tuple(df.groupby('Sport')))

target_sport = 'Basketball'
mens_data = df.loc[(df['Sex'] == "M") & (df['Sport'] == target_sport)].drop_duplicates()
features = ['Team', 'Height', 'Age', 'Weight', 'Population']
y = mens_data['Weighted_Medal']
X = mens_data[features]
X = pd.get_dummies(data=X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
regr = linear_model.LinearRegression().fit(X_train, y_train)
# Use the model to predict y from X_test
y_pred = regr.predict(X_test)
# Root mean squared error
rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
print("Linear Regression")
print('Root mean squared error (RMSE):', rmse)
print('R-squared score:', r2)





mens_data = df.loc[df['Sex'] == "M", ['Sport', 'Age','Weight', 'Height']].drop_duplicates()
womens_data = df.loc[df['Sex'] == "F", ['Sport', 'Age','Weight', 'Height']].drop_duplicates()

#all_sports = dict(tuple(df.groupby('Sport')))
sports = ['Volleyball', 'Weightlifting', 'Gymnastics']
sport_accuracy_preds = []
for sport in sports:
    this_sport = df[df['Sport']==sport]
    not_this_sport = df[df['Sport'] != sport]

    this_sport_heights = this_sport.values[:, 5]
    this_sport_weights = this_sport.values[:, 6]
    not_this_sport_heights =not_this_sport.values[:, 5]
    not_this_sport_weights = not_this_sport.values[:, 6]
    
    plt.figure(figsize=(10,10))
    title = sport+' and Non-'+sport+' Height and Weight'
    plt.title(title)
    plt.scatter(not_this_sport_weights, not_this_sport_heights, color='#0000ff', marker = '.', label = 'Non-'+sport, alpha=0.3)
    plt.scatter(this_sport_weights, this_sport_heights, color='#ff0000', marker = 'x', label = sport, alpha=0.3)
    plt.legend(loc = 'lower right')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Height (cm)')
    plt.show()
    mens_data['target'] = np.where(mens_data['Sport'] == sport, 1, 0)

    Y, X = dmatrices('target ~ 0 + Weight + Height', data = mens_data, return_type = 'dataframe')
    y = Y['target'].values

    kfold = model_selection.StratifiedKFold(n_splits = 5, shuffle = True).split(X, y)

    model = neighbors.KNeighborsClassifier(n_neighbors = 18)

    accuracies = []
    for train, holdout in kfold:
        model.fit(X.iloc[train], y[train])
        prediction_on_test = model.predict(X.iloc[holdout])
        accuracies.append(metrics.accuracy_score(y[holdout], prediction_on_test))

    print(np.mean(accuracies))
    sport_accuracy_preds.append((sport, np.mean(accuracies)))






features = ['Age', 'Height', 'Weight']
athletics['target'] = np.where(athletics['Sport'] == 'Basketball', 1, 0)
y = athletics['target']
X = athletics[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

neigh = neighbors.KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
y_pred = neigh.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("RMSE:", sqrt(mse))
print("R-squared score:", metrics.r2_score(y_test, y_pred))


clf = linear_model.LogisticRegression(solver='liblinear').fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)





athletes = df.loc[df['Sex'] == 'M', ['Sport', 'Age', 'Weight', 'Height', 'Team']].drop_duplicates()
athletes['target'] = np.where(athletes['Sport'] == 'Basketball', 1, 0)


Y, X = dmatrices('target ~ 0 + Weight + Height + Team',
                data = athletes,
                return_type = 'dataframe')

y = Y['target'].values

accuracies = []

kfold = model_selection.StratifiedKFold(n_splits = 5, shuffle = True).split(X, y)

model = neighbors.KNeighborsClassifier(n_neighbors = 20)

for train, holdout in kfold:
    model.fit(X.iloc[train], y[train])
    prediction_on_test = model.predict(X.iloc[holdout])
    accuracies.append(metrics.accuracy_score(y[holdout], prediction_on_test))

print(np.mean(accuracies))

#general trends in weight/height
#run loops for test size, n_folds etc
'''