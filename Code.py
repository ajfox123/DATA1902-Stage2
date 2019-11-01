import numpy as np
import pandas as pd
from math import sqrt
from sklearn import metrics, neighbors
from sklearn.model_selection import train_test_split
from patsy import dmatrices
import matplotlib.pyplot as plt
import statsmodels.api as sm
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import gridplot
from bokeh.models import PrintfTickFormatter

df = pd.read_csv('Output.csv')
df = df.dropna()





#Plotting height/weight of medal winners
all_sports = df['Sport'].unique()
print(len(all_sports))
for sport in all_sports:
    this_sport = df[df['Sport']==sport]

    winners = this_sport[this_sport['Medal_Won'] == 1]
    non_winners = this_sport[this_sport['Medal_Won'] != 1]

    winners_heights = winners.values[:, 5]
    winners_weights = winners.values[:, 6]
    non_winners_heights = non_winners.values[:, 5]
    non_winners_weights = non_winners.values[:, 6]

    plt.figure(figsize=(10,10))
    title = 'Medal-Winners and Non-Medal-Winners Height and Weight for '+str(sport)
    plt.title(title)
    plt.scatter(non_winners_weights, non_winners_heights, color='#0000ff', marker = '.', label = 'Non-medal winner', alpha=0.5)
    plt.scatter(winners_weights, winners_heights, color='#ff0000', marker = 'x', label = 'Medal winner', alpha=0.5)
    plt.legend(loc = 'lower right')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Height (cm)')
    plt.show()





team_events = pd.pivot_table(df,
                             index = ['Team', 'Year', 'Event'],
                             columns = 'Medal',
                             values = 'Medal_Won',
                             aggfunc = 'sum',
                             fill_value = 0).drop('No Medal', axis = 1).reset_index()
team_events = team_events.loc[team_events['Gold'] > 1, :]
team_events = team_events['Event'].unique()
print(*team_events, sep='\n')
remove_events = ["Gymnastics Men's Horizontal Bar",
                 "Gymnastics Women's Balance Beam",
                 "Swimming Women's 100 metres Freestyle",
                 "Swimming Men's 50 metres Freestyle"]
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

top_countries = medal_tally.groupby('Team')['Medal_Won_Corrected'].agg('sum').\
reset_index().sort_values('Medal_Won_Corrected', ascending = False).head(n=4)
print(top_countries)
top_countries = ['USA', 'Russia', 'Germany', 'China']
top_countries_mask = df['Team'].map(lambda x: x in top_countries)





#Plotting population vs medal tally
year_team_pop = df.loc[:, ['Year', 'Team', 'Population']].drop_duplicates()
medal_tally_pop = medal_tally.merge(year_team_pop,
                                   left_on = ['Year', 'Team'],
                                   right_on = ['Year', 'Team'],
                                   how = 'left')
correlation = medal_tally_pop.loc[:, ['Population', 'Medal_Won_Corrected']].corr()['Medal_Won_Corrected'][0]
plt.figure(figsize=(10,10))
plt.scatter(medal_tally_pop.loc[:, 'Population'],
            medal_tally_pop.loc[:, 'Medal_Won_Corrected'] ,
            marker = 'o',
            alpha = 0.5)
plt.xlabel('Country Population')
plt.ylabel('Number of Medals')
plt.title('Population versus medal tally')
plt.text(np.nanpercentile(medal_tally_pop['Population'], 98.6),
         max(medal_tally_pop['Medal_Won_Corrected']) - 50,
         "Correlation = " + str(correlation),)
plt.show()






#Plotting contingent size vs medal tally for top countries
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
for country in top_countries:
    plt.figure(figsize=(10,10))
    contingent_size[country].plot(linestyle = '-', marker = 'o', linewidth = 1, color = 'red', label = 'Contingent Size')
    year_team_medals[country].plot(linestyle = '-', marker = 'x', linewidth = 1, color = 'black', label = 'Medal Tally')
    plt.xlabel('Olympic Year')
    plt.ylabel('Contingent Size/Medal Tally')
    plt.title('Team '+country+'\nContingent Size vs Medal Tally')
    plt.legend(loc = 'best')
    plt.show()





#Correlation between contingent size vs corrected medal tally
all_countries = df['Team'].unique()
all_countries_mask = df['Team'].map(lambda x: x in all_countries)
year_team_medals = pd.pivot_table(medal_tally,
                                  index = 'Year',
                                  columns = 'Team',
                                  values = 'Medal_Won_Corrected',
                                  aggfunc = 'sum')[all_countries]
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
plt.figure(figsize=(10,10))
plt.scatter(contingent_medals.loc[:, 'Contingent'],
            contingent_medals.loc[:, 'Medal_Count'] ,
            marker = 'o',
            alpha = 0.5)
plt.xlabel('Contingent Size')
plt.ylabel('Medal Tally')
plt.title('Contingent Size versus Medal Tally')
correlation = contingent_medals.loc[:, ['Contingent', 'Medal_Count']].corr()['Medal_Count'][0]
plt.text(np.nanpercentile(contingent_medals['Contingent'], 90.6),
         max(contingent_medals['Medal_Count']) - 50,
         "Correlation = " + str(correlation),)
plt.show()





#Plotting contingent size vs population
contingent_pop = year_team_pop.merge(contingent_size_unstack,
                                     left_on = ['Year', 'Team'],
                                     right_on = ['Year', 'Team'],
                                     how = 'left')
correlation = contingent_pop.loc[:, ['Population', 'Contingent']].corr()['Contingent'][0]
plt.figure(figsize=(10,10))
plt.scatter(contingent_pop.loc[:, 'Population'],
            contingent_pop.loc[:, 'Contingent'] ,
            marker = 'o',
            alpha = 0.5)
plt.xlabel('Country Population')
plt.ylabel('Contingent Size')
plt.title('Population versus Contingent Size')
plt.text(np.nanpercentile(contingent_pop['Population'], 98.6),
         max(contingent_pop['Contingent']) - 50,
         "Correlation = " + str(correlation),)
plt.show()





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










#End of analysis





#Start of predictive models










#Getting population
year_team_pop = df.loc[:, ['Year', 'Team', 'Population']].drop_duplicates()
print(year_team_pop.head())

#Merging corrected medal count for that year
medal_pop = medal_tally.merge(year_team_pop,
                              left_on = ['Year', 'Team'],
                              right_on = ['Year', 'Team'],
                              how = 'left')
print(medal_pop.head())

#Plotting country populations and medals won as a histogram
plt.figure(figsize=(10,10))
medal_pop['Population'].hist(bins = 30)
plt.title('Population Histogram for Countries')
plt.xlabel('Population Size - Binned')
plt.ylabel('Number of Countries')
plt.show()

#Applying a logarithmic scale to population size to better fit linear model
medal_pop['Log_Population'] = np.log(medal_pop['Population'])

#Merging contingent size
medal_pop_contingent = medal_pop.merge(contingent_size_unstack,
                                       left_on = ['Year', 'Team'],
                                       right_on = ['Year', 'Team'],
                                       how = 'left')
print(medal_pop_contingent.head().to_string())





#Models using the logarithm of population and contingent size
print('Models using contingent size and log of population')
y, X = dmatrices('Medal_Won_Corrected ~ Log_Population + Contingent',
                 data = medal_pop_contingent,
                 return_type = 'dataframe')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = sm.OLS(y_train, X_train)
result = model.fit()
y_predicted = result.predict(X_test)
print('OLS')
print('Root mean squared error (RMSE):',np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
print('R-squared score:',result.rsquared)

neighbs = [0, np.inf, 0]
for i in range(1,50):
    neigh = neighbors.KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    if rmse < neighbs[1]:
        neighbs = [i, rmse, r2]
print("KNN")
print('Neighbours used:', neighbs[0])
print('Root mean squared error (RMSE):', neighbs[1])
print('R-squared score:', neighbs[2])





#Models using the logarithm of population, contingent size and team
print('\nModels using contingent size, log of population and team')
y, X = dmatrices('Medal_Won_Corrected ~ Log_Population + Contingent + Team',
                 data = medal_pop_contingent,
                 return_type = 'dataframe')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = sm.OLS(y_train, X_train)
result = model.fit()
y_predicted = result.predict(X_test)
print('OLS')
print('Root mean squared error (RMSE):',np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
print('R-squared score:',result.rsquared)

neighbs = [0, np.inf, 0]
for i in range(1,50):
    neigh = neighbors.KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    if rmse < neighbs[1]:
        neighbs = [i, rmse, r2]
print("KNN")
print('Neighbours used:', neighbs[0])
print('Root mean squared error (RMSE):', neighbs[1])
print('R-squared score:', neighbs[2])










#End of predictive models

#Start of interactive visualisations










# Output inline in the notebook
output_file('Interactive Visualisation.html', title='Interactive Visualisation')

# Store the data in a ColumnDataSource
cds = ColumnDataSource(medal_pop_contingent.loc[:, ['Contingent', 'Population', 'Medal_Won_Corrected', 'Team', 'Year']])

#Configure tools
hover = HoverTool(tooltips = [("Index", "@index"),
                              ("Team", "@Team"),
                              ("Year", "@Year"),
                              ("Population", "@Population"),
                              ("Contingent Size", "@Contingent"),
                              ("Medal Tally", "@Medal_Won_Corrected")])
toolList = [hover, 'redo,undo,click,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,tap,save']

#Draw first plot
contingent_plot = figure(title='Contingent Size vs Medal Tally',
                plot_height=650, plot_width=650, tools=toolList,
                x_axis_label='Contingent Size', y_axis_label='Medal Tally')
contingent_plot.circle(x='Contingent', y='Medal_Won_Corrected', source=cds,
              size=7, color='navy', alpha=0.4, line_color="black", line_width=1)
contingent_plot.title.align = "center"

#Draw second plot
pop_plot = figure(title='Population vs Medal Tally',
                plot_height=650, plot_width=650, tools=toolList,
                x_axis_label='Population', y_axis_label='Medal Tally')
pop_plot.square(x='Population', y='Medal_Won_Corrected', source=cds,
              size=7, color='green', alpha=0.4,line_color="black", line_width=1)
pop_plot.xaxis[0].formatter = PrintfTickFormatter(format="%4.1e")
pop_plot.title.align = "center"

# Create layout
grid = gridplot([[contingent_plot, pop_plot]])

# Visualize
show(grid)
