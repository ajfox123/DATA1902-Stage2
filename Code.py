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
from sklearn import metrics, linear_model, model_selection, neighbors, tree
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

top_countries = medal_tally.groupby('Team')['Medal_Won_Corrected'].agg('sum').reset_index().sort_values('Medal_Won_Corrected', ascending = False).head(n=4)
print(top_countries)
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

male_data = df[df['Sex'] == "M"]
female_data = df[df['Sex'] == "F"]






#Plotting height/weight for each sport vs all other athletes
all_sports = dict(tuple(df.groupby('Sport')))
specific_sports = ['Athletics', 'Basketball', 'Weightlifting','Rhythmic Gymnastics', 'Gymnastics']
sport_accuracy_preds = []
for sport in specific_sports:
    this_sport = df[df['Sport']==sport]
    winners = this_sport[this_sport['Medal_Won'] == 1]
    non_winners = this_sport[this_sport['Medal_Won'] != 1]
    winners_heights = winners.values[:, 5]
    winners_weights = winners.values[:, 6]
    non_winners_heights = non_winners.values[:, 5]
    non_winners_weights = non_winners.values[:, 6]
    
    plt.figure(figsize=(10,10))
    title = 'Medal-Winners and Non-Medal-Winners Height and Weight for',sport
    plt.title(title)
    plt.scatter(non_winners_weights, non_winners_heights, color='#0000ff', marker = '.', label = 'Non-'+sport, alpha=0.3)
    plt.scatter(winners_weights, winners_heights, color='#ff0000', marker = 'x', label = sport, alpha=0.3)
    plt.legend(loc = 'lower right')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Height (cm)')
    plt.show()

    



#Here we try to predict, using logistic regression then KNN, whether an athlete does given sport or not based on their height and weight alone
for target_sport in specific_sports:
    male_data['target'] = np.where(male_data['Sport'] == target_sport, 1, 0)
    X = df.values[:, 5:7]
    y = df.values[:, -1]
    y=y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clf = linear_model.LogisticRegression(solver='liblinear').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy for "+target_sport+" using Logistic Regression:", metrics.accuracy_score(y_test, y_pred))

for target_sport in specific_sports:
    male_data['target'] = np.where(male_data['Sport'] == target_sport, 1, 0)
    Y, X = dmatrices('target ~ 0 + Weight + Height',
                    data = male_data,
                    return_type = 'dataframe')
    y = Y['target'].values
    accuracies = []
    kfold = model_selection.StratifiedKFold(n_splits = 5, shuffle = True).split(X, y)
    model = neighbors.KNeighborsClassifier(n_neighbors = 20,
                                          p = 2,
                                          weights = 'uniform')
    for train, holdout in kfold:
        model.fit(X.iloc[train], y[train])
        prediction_on_test = model.predict(X.iloc[holdout])
        accuracies.append(metrics.accuracy_score(y[holdout], prediction_on_test))
    print("Accuracy for "+target_sport+" using K-Nearest-Neighbors classification:", np.mean(accuracies))





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





#Graphing composition of medals between top 4 countries
top_countries_medal_mask = medal_tally_agnostic['Team'].map(lambda x: x in top_countries)
medal_composition = pd.pivot_table(medal_tally_agnostic[top_countries_medal_mask],
                                     index = ['Team'],
                                     columns = 'Medal',
                                     values = 'Medal_Won_Corrected',
                                     aggfunc = 'sum',
                                     fill_value = 0).drop('No Medal', axis = 1)
medal_composition = medal_composition.loc[:, ['Gold', 'Silver', 'Bronze']]
medal_composition.plot(kind = 'bar', stacked = True, figsize = (10, 10), rot = 0)
plt.title("Composition of Top Four Countries Medals")
plt.ylabel("Medal Count")
plt.show()





#Linear Regression model for athlete winning a medal in a sport
for target_sport in specific_sports:
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


#Some statistics for height/weight/age in each sport by sex
male_stats = {'largest_avg_height':['',np.NINF],
         'lowest_avg_height':['',np.inf],
         'largest_avg_weight':['',np.NINF],
         'lowest_avg_weight':['',np.inf],
         'largest_avg_age':['',np.NINF],
         'lowest_avg_age':['',np.inf],
         'largest_std_height':['',np.NINF],
         'lowest_std_height':['',np.inf],
         'largest_std_weight':['',np.NINF],
         'lowest_std_weight':['',np.inf],
         'largest_std_age':['',np.NINF],
         'lowest_std_age':['',np.inf]
         }
female_stats = {'largest_avg_height':['',np.NINF],
         'lowest_avg_height':['',np.inf],
         'largest_avg_weight':['',np.NINF],
         'lowest_avg_weight':['',np.inf],
         'largest_avg_age':['',np.NINF],
         'lowest_avg_age':['',np.inf],
         'largest_std_height':['',np.NINF],
         'lowest_std_height':['',np.inf],
         'largest_std_weight':['',np.NINF],
         'lowest_std_weight':['',np.inf],
         'largest_std_age':['',np.NINF],
         'lowest_std_age':['',np.inf]
         }

male_sports = dict(tuple(male_data.groupby('Sport')))
for sport in male_sports:
    avg_height = round(np.mean(male_sports[sport]['Height']))
    avg_weight = round(np.mean(male_sports[sport]['Weight']))
    avg_age = round(np.mean(male_sports[sport]['Age']))
    std_height = round(np.std(male_sports[sport]['Height']))
    std_weight = round(np.std(male_sports[sport]['Weight']))
    std_age = round(np.std(male_sports[sport]['Age']))
    if avg_height > male_stats['largest_avg_height'][1]:
        male_stats['largest_avg_height'] = [sport, avg_height]
    if avg_height < male_stats['lowest_avg_height'][1]:
        male_stats['lowest_avg_height'] = [sport, avg_height]
    if avg_weight > male_stats['largest_avg_weight'][1]:
        male_stats['largest_avg_weight'] = [sport, avg_weight]
    if avg_weight < male_stats['lowest_avg_weight'][1]:
        male_stats['lowest_avg_weight'] = [sport, avg_weight]
    if avg_age > male_stats['largest_avg_age'][1]:
        male_stats['largest_avg_age'] = [sport, avg_age]
    if avg_age < male_stats['lowest_avg_age'][1]:
        male_stats['lowest_avg_age'] = [sport, avg_age]
    if std_height > male_stats['largest_std_height'][1]:
        male_stats['largest_std_height'] = [sport, std_height]
    if std_height < male_stats['lowest_std_height'][1]:
        male_stats['lowest_std_height'] = [sport, std_height]
    if std_height > male_stats['largest_std_weight'][1]:
        male_stats['largest_std_weight'] = [sport, std_weight]
    if std_weight < male_stats['lowest_std_weight'][1]:
        male_stats['lowest_std_weight'] = [sport, std_weight]
    if std_age > male_stats['largest_std_age'][1]:
        male_stats['largest_std_age'] = [sport, std_age]
    if std_age < male_stats['lowest_std_age'][1]:
        male_stats['lowest_std_age'] = [sport, std_age]
        
female_sports = dict(tuple(female_data.groupby('Sport')))
for sport in female_sports:
    avg_height = round(np.mean(female_sports[sport]['Height']))
    avg_weight = round(np.mean(female_sports[sport]['Weight']))
    avg_age = round(np.mean(female_sports[sport]['Age']))
    std_height = round(np.std(female_sports[sport]['Height']))
    std_weight = round(np.std(female_sports[sport]['Weight']))
    std_age = round(np.std(female_sports[sport]['Age']))
    if avg_height > female_stats['largest_avg_height'][1]:
        female_stats['largest_avg_height'] = [sport, avg_height]
    if avg_height < female_stats['lowest_avg_height'][1]:
        female_stats['lowest_avg_height'] = [sport, avg_height]
    if avg_weight > female_stats['largest_avg_weight'][1]:
        female_stats['largest_avg_weight'] = [sport, avg_weight]
    if avg_weight < female_stats['lowest_avg_weight'][1]:
        female_stats['lowest_avg_weight'] = [sport, avg_weight]
    if avg_age > female_stats['largest_avg_age'][1]:
        female_stats['largest_avg_age'] = [sport, avg_age]
    if avg_age < female_stats['lowest_avg_age'][1]:
        female_stats['lowest_avg_age'] = [sport, avg_age]
    if std_height > female_stats['largest_std_height'][1]:
        female_stats['largest_std_height'] = [sport, std_height]
    if std_height < female_stats['lowest_std_height'][1]:
        female_stats['lowest_std_height'] = [sport, std_height]
    if std_height > female_stats['largest_std_weight'][1]:
        female_stats['largest_std_weight'] = [sport, std_weight]
    if std_weight < female_stats['lowest_std_weight'][1]:
        female_stats['lowest_std_weight'] = [sport, std_weight]
    if std_age > female_stats['largest_std_age'][1]:
        female_stats['largest_std_age'] = [sport, std_age]
    if std_age < female_stats['lowest_std_age'][1]:
        female_stats['lowest_std_age'] = [sport, std_age]
print("Male stats:",male_stats)
print("Female stats:",female_stats)
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