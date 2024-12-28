#!/usr/bin/env python
# coding: utf-8

# # NBA Fantasy Basketball Optimizer 
# 
# This notebook aims to create a model that will be trained on past NBA player's metrics and how they progressed through successive seasons, and will effectively predict current player's performances in the coming season. The model will then find the optimized ten-player lineup that will maximize fantasy basketball points, while following certain constraints.

# ## I. Creating the cleaned dataframe
# 
# Reading in all the files (total of 10 different files across 4 sources) and cleaning up each individual dataframe before creating one final dataframe to use to train our model. 

# In[ ]:


#
# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#
import sklearn.linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, r2_score
from sklearn.model_selection import train_test_split , GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
#
from pulp import LpMaximize, LpProblem, LpVariable, lpSum


# In[ ]:


# 
# Reading in our dataframes
df_1 = pd.read_csv('common_player_info_utf.csv')   # Contains physical metrics of players
df_2 = pd.read_csv('draft_history_utf.csv')        # Contains draft statistics
#
# Choosing the columns to keep
df_1 = df_1[['display_first_last','height','weight','season_exp','rosterstatus','from_year','to_year']]
df_1 = df_1.rename(columns={'display_first_last': 'player_name'})
df_2 = df_2[['player_name','round_number','overall_pick']]
#
# Combine the two data frames
df = pd.merge(df_1,df_2)
#
# Fix the 'height' column
pattern = r'^\d{1,2}-[A-Za-z]{3}$' # Setting desired pattern
df = df[df['height'].notna() & df['height'].str.match(pattern, na=False)]  # Eliminating all dates that do not match the pattern
# Creating a function to convert the non-integer heights to inches
def convert_to_inches(value):
    # Mapping month to feet
    month_to_feet = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    # Check if the value is in month-day format
    if '-' in value:
        inches, month = value.split('-')
        inches = int(inches)  # Convert inches to integer
        feet = month_to_feet.get(month[:3], 0)  # Get the corresponding feet for the month
        #
        # Now that we have feet and inches, convert to total inches
        total_inches = feet * 12 + inches
        return total_inches
    return None 
df['height'] = df['height'].apply(convert_to_inches)
# 
# Create a mapping dictionary to one-hot-encode the 'rosterstatus' column
position_mapping = {'Active': 1,'Inactive': 2}
df.loc[:, 'rosterstatus'] = df['rosterstatus'].replace(position_mapping)
df['rosterstatus'].fillna(2,inplace=True)
#
# Replacing NA values with averages
df.fillna(df.mean(numeric_only=True),inplace=True)
df = df.dropna()
#
# Filtering out rows with over 20% of zeros
zero_percentage = (df == 0).sum(axis=1) / df.shape[1] * 100
df = df.copy()
df = df[zero_percentage <= 20]
#
# Ensuring all values are floats
# Convert all columns except for 'player_name' to float
exclude_column = 'player_name'
for col in df.columns:
    if col != exclude_column:
        df[col] = df[col].astype(int)
df[exclude_column] = df[exclude_column].astype(str)
#
# Reorder the columns
new_order = ['player_name','rosterstatus','season_exp','from_year','to_year','round_number','overall_pick','height','weight']
df = df[new_order]
#
# Print the new dataframe
print()
print(df.head(5))
df_first = df


# In[ ]:


#
# Reading in our dataframe
df = pd.read_csv('1950_to_2018.csv')   # Years 1950 to 2018
#
# Choosing the columns to keep
df = df[['Year','Player','Pos','Age','Tm','G','MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','TRB','AST','STL','BLK','TOV','PF','PTS']]
# 
# Create a mapping dictionary to one-hot-encode the 'Pos' column
position_mapping = {'G-F':2,'SG':2,'SF':1,'F':1,'G':2,'F-C':1,'PG':2,'F-G':2,'C':1,'PF':1,'C-F':1,'PF-C':1,'SF-SG':1,'C-PF':1,'SG-SF':2,
                    'PF-SF':1,'SF-PF':1,'SG-PG':2,'SF-PG':1,'C-SF':1,'PG-SG':2,'PG-SF':2,'SG-PF':2}
df = df.copy()
df.loc[:, 'Pos'] = df['Pos'].replace(position_mapping)
#
# Remove rows where the number of NaN values exceeds the threshold
df = df[df.isna().sum(axis=1) <= 0.2*df.shape[1]]
#
# Replacing NaN values with zero
df.fillna(0,inplace=True)
#
# Filtering out rows with over 20% of zeros
zero_percentage = (df == 0).sum(axis=1) / df.shape[1] * 100
df = df.copy()
df = df[zero_percentage <= 20]
#
# Convert all columns except for 'player_name' to integers
exclude_column = ['Year','Pos','Age','G','GS','MP','FG','FGA','3P','3PA','2P','2PA']
for col in df.columns:
    if col in exclude_column:
        df[col] = df[col].astype(int)
#
# Convert total stats to per-game stats
columns_to_convert = ['MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','TRB','AST','STL','BLK','TOV','PF','PTS']
for col in df.columns:
    if col in columns_to_convert:
        df[col] = df[col] / df['G']
#
# Print the new dataframe
print()
print(df.head(5))
df_second = df


# In[ ]:


#
# Reading in our dataframes
df = pd.read_csv('2018to2022.csv')   # Years 2018 to 2021
# 
# Reordering and renaming the columns
df = df[['Player','Pos','Age','Team','Games','Minutes Played','Fields Goal','Fields Goal Attempted','3-points Field Goal',
         '3-points Field Goal Attempted','2-points Field Goal','2-points Field Goal Attempted','Free Throws','Free Throws Attempted',
         'Total Rebounds','Assists','Steals','Blocks','Turnovers','Personal Fouls','Points','Year']]
new_order = ['Year','Player','Pos','Age','Team','Games','Minutes Played','Fields Goal','Fields Goal Attempted','3-points Field Goal',
         '3-points Field Goal Attempted','2-points Field Goal','2-points Field Goal Attempted','Free Throws','Free Throws Attempted',
         'Total Rebounds','Assists','Steals','Blocks','Turnovers','Personal Fouls','Points']
df = df[new_order]
df.columns = ['Year','Player','Pos','Age','Tm','G','MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','TRB','AST','STL','BLK','TOV','PF','PTS']
# 
# Create a mapping dictionary to one-hot-encode the 'Pos' column
position_mapping = {'C':1,'PF':1,'SG':2,'PG':2,'SF':1,'SG-PG':2,'SG-SF':2,'PF-SF':1,'PG-SG':2,'SF-SG':1,'C-PF':1,'PF-C':1,'SF-PF':1,'SF-C':1,'SG-PF':2}
df = df.copy()
df.loc[:, 'Pos'] = df['Pos'].replace(position_mapping)
#
# Remove rows where the number of NaN values exceeds the threshold
df = df[df.isna().sum(axis=1) <= 0.2*df.shape[1]]
#
# Replacing NaN values with zero
df.fillna(0,inplace=True)
#
# Filtering out rows with over 20% of zeros
zero_percentage = (df == 0).sum(axis=1) / df.shape[1] * 100
df = df.copy()
df = df[zero_percentage <= 20]
#
# Print the new dataframe
print()
print(df.head(5))
df_third = df


# In[ ]:


#
# Reading in our dataframes
regular_2021to2022 = pd.read_csv('2021to2022_regular.csv', sep=';')     # Years 2021 to 2022
playoffs_2021to2022 = pd.read_csv('2021to2022_playoffs.csv', sep=';')   # Years 2021 to 2022
regular_2022to2023 = pd.read_csv('2022to2023_regular.csv', sep=';')     # Years 2022 to 2023
playoffs_2022to2023 = pd.read_csv('2022to2023_playoffs.csv', sep=';')   # Years 2022 to 2023
regular_2023to2024 = pd.read_csv('2023to2024_regular.csv', sep=';')     # Years 2023 to 2024
playoffs_2023to2024 = pd.read_csv('2023to2024_playoffs.csv', sep=';')   # Years 2023 to 2024
#
# Combining the dataframes
df_1 = pd.concat([regular_2021to2022, playoffs_2021to2022], ignore_index=True)
df_1['Year'] = 2022
df_2 = pd.concat([regular_2022to2023, playoffs_2022to2023], ignore_index=True)
df_2['Year'] = 2023
df_3 = pd.concat([regular_2023to2024, playoffs_2023to2024], ignore_index=True)
df_3['Year'] = 2024
df = pd.concat([df_1, df_2, df_3], ignore_index=True)
#
# Choosing the columns to keep
df = df[['Year','Player','Pos','Age','Tm','G','MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','TRB','AST','STL','BLK','TOV','PF','PTS']]
# 
# Create a mapping dictionary to one-hot-encode the 'Pos' column
position_mapping = {'C':1,'PF':1,'SG':2,'PG':2,'SF':1,'SG-PG':2,'SG-SF':2,'PF-SF':1,'PG-SG':2,'SF-SG':1,'C-PF':1,'PF-C':1,'SF-PF':1,'SG-PF':2}
df = df.copy()
df.loc[:, 'Pos'] = df['Pos'].replace(position_mapping)
#
# Remove rows where the number of NaN values exceeds the threshold
df = df[df.isna().sum(axis=1) <= 0.2*df.shape[1]]
#
# Replacing NaN values with zero
df.fillna(0,inplace=True)
#
# Filtering out rows with over 20% of zeros
zero_percentage = (df == 0).sum(axis=1) / df.shape[1] * 100
df = df.copy()
df = df[zero_percentage <= 20]
#
# Print the new dataframe
print()
print(df.head(5))
df_fourth = df


# In[ ]:


#
# Creating our final cleaned dataframe 
df = pd.concat([df_second, df_third, df_fourth], ignore_index=True)
label_encoder = LabelEncoder()
df['Tm'] = label_encoder.fit_transform(df['Tm'])
#
# Merge the big dataframe with the first one and drop the common column
merged_df = pd.merge(df, df_first, left_on='Player', right_on='player_name', how='left')
merged_df.drop(columns=['player_name'], inplace=True)
#
# Fill NaN values with the column mean
merged_df.fillna(merged_df.mean(numeric_only=True),inplace=True)
#
# Print the new dataframe
df = merged_df
print(df.head(5))


# ## II. Visualizing the data
# 
# Creating various different graphs to physically display the information within the dataframe. 

# In[ ]:


print("Information about our dataframe:")
print()
df.info()


# In[ ]:


df.describe(include= np.number)


# In[ ]:


print("NBA Fantasy Basketball awards points based on points scored, rebounds, assists, blocks, steals, and turnovers.")
print("Here is the distribution for each:")
print()
#
# Define relationships
histograms = [
    ('PTS', 'Total Points'),
    ('3P', '3-Point Goals'),
    ('2P', 'Field Goals'),
    ('FT', 'Free Throws'),
    ('TRB', 'Rebounds'),
    ('AST', 'Assists'),
    ('BLK', 'Blocks'),
    ('STL', 'Steals'),
    ('TOV', 'Turnovers')
]
#
# Create subplots 
fig = make_subplots(rows=3, cols=3, subplot_titles=[title for _, title in histograms])
#
# Add traces to subplots
for idx, (x_col, title) in enumerate(histograms, start=1):
    row = (idx - 1) // 3 + 1
    col = (idx - 1) % 3 + 1
    # Add histogram trace
    fig.add_trace(go.Histogram(x=df[x_col], name=title), row=row, col=col)
    
    fig.update_xaxes(title_text=x_col, row=row, col=col)
    fig.update_yaxes(title_text='Frequency', row=row, col=col)
#
# Update the layout
fig.update_layout(
    title_text='Basketball Metrics Distribution',
    height=800,
    width=1000,
    template='plotly_dark',
    showlegend=False 
)
#
# Display the plot
fig.write_image('metrics_distribution.png')


# In[ ]:


radar_columns = ['PTS','3P','2P','FT','TRB','AST','BLK','STL','TOV']

selected_players = df.sample(n=5)
fig_radar = go.Figure()

for index, player in selected_players.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[player[column] for column in radar_columns],
        theta=radar_columns,
        fill='toself',
        name=player['Player']
    ))

fig_radar.update_layout(
    title='Player Comparison - Overall Performance',
    template='plotly_dark',
    polar=dict(
        radialaxis=dict(visible=True, range=[0,15]),
    ),
)
fig_radar.write_image('radar.png')
## Discovered how to create this graph from a notebook using one of my datasets on Kaggle (credit in report)


# In[ ]:


#
# Define relationships
relationships = [('Age', 'PTS'),('PTS', 'G'),('FG', 'FGA'),('2P', '2PA'),('3P', '3PA'),('FT', 'FTA'),('TRB', 'PTS'),
                 ('STL', 'BLK'),('PF', 'BLK'),('AST', 'PTS'),('height', 'weight'),('overall_pick', 'PTS')]
#
# Create subplots
fig = make_subplots(rows=4, cols=3)
#
# Add traces to subplots
for idx, (x_col, y_col) in enumerate(relationships, start=1):
    row = (idx - 1) // 3 + 1
    col = (idx - 1) % 3 + 1
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers' if idx != 3 else 'lines'), row=row, col=col)
    
    fig.update_xaxes(title_text=x_col, row=row, col=col)
    fig.update_yaxes(title_text=y_col, row=row, col=col)
#
# Update the layout
fig.update_layout(
    title_text='Relationships between Different Columns',
    height=800,
    width=1000,
    template='plotly_dark'
)
#
# Update legend and annotations
fig.update_layout(showlegend=False)
fig.update_annotations(dict(text='', showarrow=False))
#
# Display the plot
fig.write_image('relationships.png')
## Discovered how to create this graph from a notebook using one of my datasets on Kaggle (credit in report)


# In[ ]:


#
import plotly.graph_objects as go

df['Defense'] = df['BLK'] + df['STL']
best_defending_players = df.sort_values(by='Defense', ascending=False).head(10)
#
fig_defending = go.Figure()
fig_defending.add_trace(go.Bar(x=best_defending_players['Player'], 
                               y=best_defending_players['Defense']))
fig_defending.update_layout(
    title='Top 10 Best Defending Players',
    xaxis_title='Player Name',
    yaxis_title='Defensive Performance (Combined Blocks and Steals)',
    height=500,
    width=1000,
    template='plotly_dark'
)
fig_defending.write_image('defending.png')
## Discovered how to create this graph from a notebook using one of my datasets on Kaggle (credit in report)


# In[ ]:


#
df.drop(columns=['Defense'], inplace=True)
#
best_attacking_players = df.sort_values(by='PTS', ascending=False).head(10)
#
fig_attacking = go.Figure()
fig_attacking.add_trace(go.Bar(x=best_attacking_players['Player'], y=best_attacking_players['PTS']))
#
fig_attacking.update_layout(
    title='Top 10 Best Attacking Players',
    xaxis_title='Player Name',
    yaxis_title='Total Points',
    height=500,
    width=1000,
    template='plotly_dark'
)
fig_attacking.write_image('attacking.png')
## Discovered how to create this graph from a notebook using one of my datasets on Kaggle (credit in report)


# In[ ]:


#
correlation_matrix = df.corr(numeric_only=True)
#
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,          
))
#
fig.update_layout(
    title='Correlation Heatmap',
    xaxis_title='Features',
    yaxis_title='Features',
    height=1000,
    template='plotly_dark'
)
#
fig.write_image('correlation_matrix.png')


# ## III. Choosing a model that will predict future performances
# 
# Testing various methods to determine the best model to predict future NBA players' performances. The models that will be tested are listed below. I am attempting to predict 9 different metrics: total points, 3-point goals, field goals, free throws, rebounds, assists, blocks, steals, turnovers. In the interest of time, I will train each of the three models to predict total points scored per game and choose the most effective of the three models to predict the other 8 metrics. 
# - (a) Linear Regression Model
# - (b) RandomForest Model
# - (c) Time-Series Forecasting

# In[ ]:


# 
# Creating initial X and y datasets for models
df_copy = df.copy()
df_copy['target'] = df_copy.groupby('Player')['PTS'].shift(-1)
df_copy = df_copy.dropna()
X = df_copy[['Year','Pos','Age','Tm','G','MP','FGA','3PA','2PA','FTA','TRB','AST','STL','BLK','TOV','PF','season_exp','round_number',
       'overall_pick','height','weight']]
y = df_copy['target']
#
# Splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# ### (a) Linear Regression Model

# In[ ]:


#
# Initialize and train the model
lr_model = sklearn.linear_model.LinearRegression()
lr_model.fit(X_train, y_train)
#
# Predict points using trained model
y_pred = lr_model.predict(X_test)
#
# Creating a residual plot
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Actual vs Predicted Points for Linear Regression Model')
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)
plt.text(0.5, -0.2, f'MSE: {mse:.4f}\nR2: {r2:.4f}', 
    transform=ax.transAxes, ha='center', va='top', fontsize=10,
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
plt.savefig('residual_plot_1.png', format='png')
plt.close()
#
# Print out mean squared error and R2-score
print("Mean squared error",mean_squared_error(y_test,y_pred))
print("R2-score:",r2_score(y_test, y_pred))


# The linear regression model does a fine job at predicting the total amount of points an NBA player will score in the upcoming season. We will continue on with the RandomForest model.

# ### (b) RandomForest Model

# In[ ]:


#
# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
#
# Predict points using trained model
y_pred = rf_model.predict(X_test)
#
# Creating a residual plot
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Actual vs Predicted Points for RandomForestRegression Model')
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)
plt.text(0.5, -0.2, f'MSE: {mse:.4f}\nR2: {r2:.4f}', 
    transform=ax.transAxes, ha='center', va='top', fontsize=10,
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
plt.savefig('residual_plot_2.png', format='png')
plt.close()
#
# Print out mean squared error and R2-score
print("Mean squared error",mean_squared_error(y_test,y_pred))
print("R2-score:",r2_score(y_test, y_pred))


# A RandomForest model performs very similarly to a Linear Regression model. With a little more test tuning to determine the ideal test size and random state, as well as hypertuning the model with a param_grid of n_estimators and max_depth, this model could become even more accurate, which we will do if we determine this model is the most effective type.
# 

# ### (c) Time-Series Forecasting

# In[ ]:


#
# Prepare our target
df_copy = df.copy()
df_copy['target'] = df_copy.groupby('Player')['PTS'].shift(-1)
df_copy = df_copy.dropna(subset=['target'])
#
# Initialize scalers and label encoder
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
label_encoder = LabelEncoder()
#
# Scale the features
features_to_scale = ['Year','Pos','Age','Tm','G','MP','FGA','3PA','2PA','FTA','TRB','AST','STL','BLK','TOV','PF','season_exp','round_number',
                     'overall_pick','height','weight']
df_copy[features_to_scale] = feature_scaler.fit_transform(df_copy[features_to_scale])
df_copy['target'] = target_scaler.fit_transform(df_copy[['target']])
#
# Encode the player names
df_copy['Player'] = label_encoder.fit_transform(df_copy['Player'])
#
# Prepare sequences for LSTM 
sequence_length = 5  # Use the last 5 seasons to predict the next
X = []
y = []
#
# Loop through each player
for player in df_copy['Player'].unique():
    player_data = df_copy[df_copy['Player'] == player]
    for i in range(sequence_length, len(player_data)):
        X.append(player_data.iloc[i-sequence_length:i][features_to_scale].values)
        y.append(player_data.iloc[i]['target'])
#
# Convert to numpy arrays
X = np.array(X)
y = np.array(y)
#
# Reshape data to [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
#
# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
#
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
#
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)
#
# Make predictions
y_pred = model.predict(X_test)
#
# Inverse transform the predictions and actual values
y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)) 
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)) 
#
# Plot the predictions vs actual data
plt.plot(y_test_rescaled, label='Actual Points')
plt.plot(y_pred_rescaled, label='Predicted Points')
plt.legend()
plt.title('Actual vs Predicted Points for Time-Series Forecasting Model')
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)
plt.text(0.5, -0.2, f'MSE: {mse:.4f}\nR2: {r2:.4f}', 
    transform=ax.transAxes, ha='center', va='top', fontsize=10,
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))plt.savefig('residual_plot_3.png', format='png')
plt.close()
#
# Print out mean squared error and R2-score
print("Mean squared error:", mean_squared_error(y_test_rescaled, y_pred_rescaled))
print("R2-score:", r2_score(y_test_rescaled, y_pred_rescaled))


# Time-series forcasting using a Long Short-Term Memory model is more effective at predicting future points than the Linear Regression model, but not our RandomForestRegression model. Now, we will hypertune our RandomForestRegression model to find the best parameters, and then use the model to predict current players' performances in the coming season.

# ## IV. Hypertuning our model to make predictions
# 
# Here, we are using GridSearchCV to loop over a variety of parameters to determine which ones minimize our mean squared error. The best parameters will then be used to train the model and predict a variety of different metrics.

# In[ ]:


#
# Prepare our target values
df_copy = df.copy()
df_copy['target_PTS'] = df_copy.groupby('Player')['PTS'].shift(-1)
df_copy['target_3P'] = df_copy.groupby('Player')['3P'].shift(-1)
df_copy['target_2P'] = df_copy.groupby('Player')['2P'].shift(-1)
df_copy['target_FT'] = df_copy.groupby('Player')['FT'].shift(-1)
df_copy['target_TRB'] = df_copy.groupby('Player')['TRB'].shift(-1)
df_copy['target_AST'] = df_copy.groupby('Player')['AST'].shift(-1)
df_copy['target_BLK'] = df_copy.groupby('Player')['BLK'].shift(-1)
df_copy['target_STL'] = df_copy.groupby('Player')['STL'].shift(-1)
df_copy['target_TOV'] = df_copy.groupby('Player')['TOV'].shift(-1)
df_copy = df_copy.dropna()
#
# Initialize scalers and label encoder
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
label_encoder = LabelEncoder()
#
# Scale the features
features_to_scale = ['Year','Pos','Age','G','Tm','MP','FGA','3P','3PA','2P','2PA','FT','FTA','TRB','AST',
                     'STL','BLK','TOV','PF','season_exp','round_number','overall_pick','height','weight','PTS']
df_copy[features_to_scale] = feature_scaler.fit_transform(df_copy[features_to_scale])
df_copy['target_PTS'] = target_scaler.fit_transform(df_copy[['target_PTS']])
df_copy['target_3P'] = target_scaler.fit_transform(df_copy[['target_3P']])
df_copy['target_2P'] = target_scaler.fit_transform(df_copy[['target_2P']])
df_copy['target_FT'] = target_scaler.fit_transform(df_copy[['target_FT']])
df_copy['target_TRB'] = target_scaler.fit_transform(df_copy[['target_TRB']])
df_copy['target_AST'] = target_scaler.fit_transform(df_copy[['target_AST']])
df_copy['target_BLK'] = target_scaler.fit_transform(df_copy[['target_BLK']])
df_copy['target_STL'] = target_scaler.fit_transform(df_copy[['target_STL']])
df_copy['target_TOV'] = target_scaler.fit_transform(df_copy[['target_TOV']])
#
# Encode the player names
df_copy['Player'] = label_encoder.fit_transform(df_copy['Player'])
#
X = df_copy[['Year','Pos','Age','Tm','G','MP','FGA','3PA','2PA','FTA','TRB','AST','STL','BLK','TOV','PF','season_exp','round_number',
       'overall_pick','height','weight']]
y = df_copy[['target_PTS','target_3P','target_2P','target_FT','target_TRB','target_AST','target_BLK','target_STL','target_TOV']]
#
# Splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#
# Example param_grid
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],       # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum number of samples per leaf
    'max_features': ['auto', 'sqrt'],      # Number of features to consider at each split
    'bootstrap': [True, False]             # Whether to use bootstrapping
}
#
# RandomForestRegressor model
rf = RandomForestRegressor(random_state=42)
#
# GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_result = grid_search.fit(X_train, y_train)


# In[ ]:


#
# Test our best model
y_pred_actual = grid_result.best_estimator_.predict(X_test)
#
# Predicted values
predicted_PTS = y_pred_actual[:,0]
predicted_3P = y_pred_actual[:,1]
predicted_2P = y_pred_actual[:,2]
predicted_FT = y_pred_actual[:,3]
predicted_TRB = y_pred_actual[:,4]
predicted_AST = y_pred_actual[:,5]
predicted_BLK = y_pred_actual[:,6]
predicted_STL = y_pred_actual[:,7]
predicted_TOV = y_pred_actual[:,8]
#
# Actual values
y_test_actual = y_test
actual_PTS = y_test_actual.iloc[:,0]
actual_3P = y_test_actual.iloc[:,1]
actual_2P = y_test_actual.iloc[:,2]
actual_FT = y_test_actual.iloc[:,3]
actual_TRB = y_test_actual.iloc[:,4]
actual_AST = y_test_actual.iloc[:,5]
actual_BLK = y_test_actual.iloc[:,6]
actual_STL = y_test_actual.iloc[:,7]
actual_TOV = y_test_actual.iloc[:,8]
#
# List of predictions and actual values for each statistic
predictions = {
    "PTS": (predicted_PTS, actual_PTS),
    "3P": (predicted_3P, actual_3P),
    "2P": (predicted_2P, actual_2P),
    "FT": (predicted_FT, actual_FT),
    "TRB": (predicted_TRB, actual_TRB),
    "AST": (predicted_AST, actual_AST),
    "BLK": (predicted_BLK, actual_BLK),
    "STL": (predicted_STL, actual_STL),
    "TOV": (predicted_TOV, actual_TOV)
}
#
# Create prediction plots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for idx, (stat, (predicted, actual)) in enumerate(predictions.items()):
    ax = axes[idx]
    ax.scatter(actual, predicted, label='Predicted vs Actual', color='blue')
    ax.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linewidth=2, label='Ideal Fit')
    #
    # Add labels and title
    ax.set_title(f'{stat} Prediction vs Actual')
    ax.set_xlabel(f'Actual {stat}')
    ax.set_ylabel(f'Predicted {stat}')
    #
    # Calculate Mean Squared Error and R2-score
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    #
    # Add MSE and R2-score as text below each plot
    ax.text(0.5, -0.2, f'MSE: {mse:.4f}\nR2: {r2:.4f}', 
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
    #
    # Add a legend
    ax.legend()
#
# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.savefig('all_predictions_plot.png', format='png')
plt.close()
#
# Print out the results
model = grid_result.best_estimator_
print("Best parameters: ",grid_result.best_params_)
print("Best score: ", grid_result.best_score_)


# In[ ]:


# 
# Selecting current players playing in the upcoming season
df_copy = df.copy()
current_players = df_copy[df_copy['rosterstatus'] == 1]
#
# Create target columns for the 2025 season (target for next season)
current_players['target_PTS'] = current_players.groupby('Player')['PTS'].shift(-1)
current_players['target_3P'] = current_players.groupby('Player')['3P'].shift(-1)
current_players['target_2P'] = current_players.groupby('Player')['2P'].shift(-1)
current_players['target_FT'] = current_players.groupby('Player')['FT'].shift(-1)
current_players['target_TRB'] = current_players.groupby('Player')['TRB'].shift(-1)
current_players['target_AST'] = current_players.groupby('Player')['AST'].shift(-1)
current_players['target_BLK'] = current_players.groupby('Player')['BLK'].shift(-1)
current_players['target_STL'] = current_players.groupby('Player')['STL'].shift(-1)
current_players['target_TOV'] = current_players.groupby('Player')['TOV'].shift(-1)
current_players = current_players.dropna(subset=['target_PTS', 'target_3P', 'target_2P', 'target_FT', 'target_TRB', 
                                                 'target_AST', 'target_BLK', 'target_STL', 'target_TOV'])
#
# Scale the features (only for the current players)
features_to_scale = ['Year','Pos','Age','G','Tm','MP','FGA','3P','3PA','2P','2PA','FT','FTA','TRB','AST',
                     'STL','BLK','TOV','PF','season_exp','round_number','overall_pick','height','weight','PTS']
current_players[features_to_scale] = feature_scaler.fit_transform(current_players[features_to_scale])
current_players['target_PTS'] = target_scaler.fit_transform(current_players[['target_PTS']])
current_players['target_3P'] = target_scaler.fit_transform(current_players[['target_3P']])
current_players['target_2P'] = target_scaler.fit_transform(current_players[['target_2P']])
current_players['target_FT'] = target_scaler.fit_transform(current_players[['target_FT']])
current_players['target_TRB'] = target_scaler.fit_transform(current_players[['target_TRB']])
current_players['target_AST'] = target_scaler.fit_transform(current_players[['target_AST']])
current_players['target_BLK'] = target_scaler.fit_transform(current_players[['target_BLK']])
current_players['target_STL'] = target_scaler.fit_transform(current_players[['target_STL']])
current_players['target_TOV'] = target_scaler.fit_transform(current_players[['target_TOV']])
#
# Encode player names
current_players['Player'] = label_encoder.fit_transform(current_players['Player'])
#
X = current_players[['Year','Pos','Age','Tm','G','MP','FGA','3PA','2PA','FTA','TRB','AST','STL','BLK','TOV','PF','season_exp','round_number',
       'overall_pick','height','weight']]
y = current_players[['target_PTS','target_3P','target_2P','target_FT','target_TRB','target_AST','target_BLK','target_STL','target_TOV']]
#
# Fit the model and make predictions
model.fit(X, y)
y_pred = model.predict(X)
y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 9))
#
# Adding predictions to a new dataframe for optimization later
predictions = []
for player in current_players['Player'].unique():
    player_data = current_players[current_players['Player'] == player]
    player_data_sorted = player_data.sort_values(by='Year', ascending=False)
    most_recent_data = player_data_sorted[player_data_sorted['Year'] == 2024]
    if not most_recent_data.empty:
        player_pos = most_recent_data['Pos'].values[0]
    else:
        player_pos = 0
    last_prediction = y_pred[player_data.shape[0] - 1]
    player_predictions = {
        'Player': player,
        'Pos': player_pos,
        'predicted_PTS': last_prediction[0],
        'predicted_3P': last_prediction[1],
        'predicted_2P': last_prediction[2],
        'predicted_FT': last_prediction[3],
        'predicted_TRB': last_prediction[4],
        'predicted_AST': last_prediction[5],
        'predicted_BLK': last_prediction[6],
        'predicted_STL': last_prediction[7],
        'predicted_TOV': last_prediction[8]
    }
    predictions.append(player_predictions)
predictions_df = pd.DataFrame(predictions)
#
# Add the most recent teams (only for the 2024 season)
most_recent_team = current_players.sort_values(by=['Player', 'Year'], ascending=[True, False]) \
                                   .drop_duplicates(subset='Player', keep='first')[['Player', 'Tm']]
predictions_df = pd.merge(predictions_df, most_recent_team, on='Player', how='left')
#
# Shift columns back 
columns_to_shift = ['predicted_PTS','predicted_3P','predicted_2P','predicted_FT','predicted_TRB','predicted_AST','predicted_BLK','predicted_STL','predicted_TOV']
predictions_df[columns_to_shift] = predictions_df[columns_to_shift].shift(1)
predictions_df = predictions_df.iloc[1:].reset_index(drop=True)
#
# Inverse transform the player names
predictions_df['Player'] = label_encoder.inverse_transform(predictions_df['Player'])
#
# Print out the predictions for the players
print(predictions_df.head(5))


# ## V. Using our predictions to optimize our fantasy basketball lineup
# 
# Our model has predicted the performances for each player next season. We will now select a lineup based on these predictions that, in theory, should maximize the amount of points the ideal fantasy basketball team lineup would get.

# In[ ]:


#
# Scoring system
scoring = {'point': 1,'three_point': 0,'field_goal': 0,'free_throw': 0,'rebound': 1.2,'assist': 1.5,'block': 3,'steal': 3,'turnover': -1}
#
# Finding a player's full predicted fantasy score for any given game
predictions_df['Total_Score'] = (
    predictions_df['predicted_PTS'] * scoring['point'] +
    predictions_df['predicted_3P'] * scoring['three_point'] +
    predictions_df['predicted_2P'] * scoring['field_goal'] +
    predictions_df['predicted_FT'] * scoring['free_throw'] +
    predictions_df['predicted_TRB'] * scoring['rebound'] +
    predictions_df['predicted_AST'] * scoring['assist'] +
    predictions_df['predicted_BLK'] * scoring['block'] +
    predictions_df['predicted_STL'] * scoring['steal'] +
    predictions_df['predicted_TOV'] * scoring['turnover']
)
#
# List of all salary data
salary_data = [
    ["Nikola Jokic", 20.6], ["Luka Doncic", 18.9], ["Joel Embiid", 17.9], ["Giannis Antetokounmpo", 17.7],
    ["Anthony Davis", 17.2], ["Shai Gilgeous-Alexander", 17.2], ["Victor Wembanyama", 17.0], ["Domantas Sabonis", 16.9],
    ["Jayson Tatum", 16.7], ["LeBron James", 16.5], ["Trae Young", 16.0], ["Donovan Mitchell", 16.0], 
    ["Kevin Durant", 16.0], ["Stephen Curry", 15.9], ["Anthony Edwards", 15.9], ["De'Aaron Fox", 15.6],
    ["Kyrie Irving", 15.5], ["Jalen Brunson", 15.4], ["Tyrese Haliburton", 15.3], ["Alperen Sengun", 15.1],
    ["LaMelo Ball", 15.1], ["Kawhi Leonard", 15.0], ["Bam Adebayo", 15.0], ["Damian Lillard", 15.0],
    ["Julius Randle", 15.0], ["Tyrese Maxey", 15.0], ["Devin Booker", 15.0], ["Scottie Barnes", 15.0],
    ["Dejounte Murray", 14.9], ["Ja Morant", 14.9], ['James Harden', 14.8], ['Paolo Banchero', 14.5], 
    ['Lauri Markkanen', 14.5], ['Desmond Bane', 14.5], ['Zion Williamson', 14.4], ['Fred VanVleet', 14.4], 
    ['DeMar DeRozan', 14.4], ['Cade Cunningham', 14.4], ['Karl-Anthony Towns', 14.2], ['Kristaps Porzingis', 14.0], 
    ['Nikola Vucevic', 14.0], ['Pascal Siakam', 14.0], ['Jaren Jackson Jr.', 14.0], ['Jaylen Brown', 13.9], ['Jamal Murray', 13.9],
    ['Jimmy Butler', 13.9], ['Rudy Gobert', 13.9], ['Paul George', 13.9], ['Jalen Johnson', 13.7],
    ['Jarrett Allen', 13.5], ['Brandon Ingram', 13.5], ['Chet Holmgren', 13.5], ['Miles Bridges', 13.5], 
    ['Evan Mobley', 13.4], ['Kyle Kuzma', 13.4], ['Franz Wagner', 13.3], ['Jalen Williams', 13.2], ['CJ McCollum', 13.0], 
    ['Deandre Ayton', 13.0], ['RJ Barrett', 13.0], ["Terry Rozier", 12.9], ["Anfernee Simons", 12.9], 
    ["Darius Garland", 12.6], ["Derrick White", 12.5], ["Coby White", 12.5], ["Tyler Herro", 12.5], 
    ["Jerami Grant", 12.5], ["Zach LaVine", 12.4], ["Jalen Green", 12.4], ["D'Angelo Russell", 12.4],
    ["Nic Claxton", 12.4], ["Mikal Bridges", 12.4], ["Myles Turner", 12.4], ["Jusuf Nurkic", 12.4], 
    ["Tobias Harris", 12.4], ["Jalen Duren", 12.4], ["Bradley Beal", 12.3], ["Austin Reaves", 12.0], 
    ["Devin Vassell", 12.0], ["Immanuel Quickley", 12.0], ["Jakob Poeltl", 12.0], ["Collin Sexton", 12.0], 
    ["Mark Williams", 12.0], ["Clint Capela", 11.9], ["Daniel Gafford", 11.9], ["Cam Thomas", 11.9], 
    ["Deni Avdija", 11.9], ["Aaron Gordon", 11.5], ["Michael Porter Jr.", 11.5], ["Draymond Green", 11.5], 
    ["John Collins", 11.5], ["Malcolm Brogdon", 11.5], ["Jrue Holiday", 11.4], ["Khris Middleton", 11.4], 
    ["Tyus Jones", 11.4], ["Caris LeVert", 11.3], ["Ivica Zubac", 11.2], ["Bogdan Bogdanovic", 11.0], 
    ["Josh Giddey", 11.0], ["Jabari Smith Jr.", 11.0], ["Brook Lopez", 11.0], ["Keegan Murray", 11.0], 
    ["Keldon Johnson", 11.0], ["Jordan Clarkson", 11.0], ["Jordan Poole", 11.0], ["Brandon Miller", 11.0], 
    ["Jonas Valanciunas", 10.9], ["Josh Hart", 10.6], ["Trey Murphy III", 10.5], ["Dennis Schroder", 10.5], 
    ["OG Anunoby", 10.5], ["Malik Monk", 10.5], ["Marcus Smart", 10.5], ["Donte DiVincenzo", 10.4], 
    ["Max Strus", 10.0], ["P.J. Washington", 10.0], ["Russell Westbrook", 10.0], ["Amen Thompson", 10.0],
    ["Tari Eason", 10.0], ["Bobby Portis", 10.0], ["Mitchell Robinson", 10.0], ["Chris Paul", 10.0], 
    ["Isaiah Hartenstein", 10.0], ["Walker Kessler", 10.0], ["Jaden Ivey", 10.0], ["Klay Thompson", 9.9], 
    ["Jonathan Kuminga", 9.9], ["Jaime Jaquez Jr.", 9.9], ["Mike Conley", 9.9], ["Scoot Henderson", 9.9], 
    ["Alex Caruso", 9.9], ["Onyeka Okongwu", 9.5], ["Herbert Jones", 9.5], ["Ayo Dosunmu", 9.5], 
    ["Dereck Lively II", 9.5], ["Buddy Hield", 9.5], ["De'Anthony Melton", 9.5], ["Andrew Wiggins", 9.5], 
    ["Cameron Johnson", 9.5], ["Jalen Suggs", 9.5], ["Wendell Carter Jr.", 9.5], ["Bennedict Mathurin", 9.5], 
    ["Aaron Nesmith", 9.5], ["Tre Jones", 9.5], ["Zach Collins", 9.5], ["Kelly Olynyk", 9.5], 
    ["Vince Williams Jr.", 9.5], ["Santi Aldama", 9.5], ["Saddiq Bey", 9.5], ["Ausar Thompson", 9.5], 
    ["Isaiah Stewart", 9.5], ["Nick Richards", 9.5], ["De'Andre Hunter", 9.4], ["Al Horford", 9.4], 
    ["Brandin Podziemski", 9.4], ["Rui Hachimura", 9.4], ["Naz Reid", 9.4], ["Ben Simmons", 9.4], 
    ["T.J. McConnell", 9.4], ["Kelly Oubre Jr.", 9.4], ["Andre Drummond", 9.4], ["Grayson Allen", 9.4], 
    ["Jeremy Sochan", 9.4], ["Brandon Clarke", 9.4], ["Marvin Bagley III", 9.4], ["Patrick Williams", 9.0], 
    ["Bojan Bogdanovic", 9.0], ["Precious Achiuwa", 9.0], ["Kevin Huerter", 9.0], ["Bruce Brown", 9.0], 
    ["Keyonte George", 9.0], ["GG Jackson", 9.0], ["Tre Mann", 9.0], ["Cody Martin", 9.0], ["Jalen Smith", 8.9], 
    ["Spencer Dinwiddie", 8.9], ["Dillon Brooks", 8.9], ["Duncan Robinson", 8.9], ["Gary Trent Jr.", 8.9], 
    ["Kentavious Caldwell-Pope", 8.9], ["Kyle Lowry", 8.9], ["Caleb Martin", 8.9], ["Luguentz Dort", 8.9], 
    ["Tim Hardaway Jr.", 8.9], ["Paul Reed", 8.9], ["Cole Anthony", 8.8], ["Norman Powell", 8.5], 
    ["Royce O'Neale", 8.5], ["Gordon Hayward", 8.5], ["Luke Kennard", 8.5], ["Alexandre Sarr", 8.5], 
    ["Grant Williams", 8.5], ["Zaccharie Risacher", 8.4], ["Talen Horton-Tucker", 8.4], ["Trayce Jackson-Davis", 8.4], 
    ["Kyle Anderson", 8.4], ["Cam Whitmore", 8.4], ["Kevin Love", 8.4], ["Dalano Banton", 8.4], ["Corey Kispert", 8.4], 
    ["Jabari Walker", 8.3], ["Payton Pritchard", 8.0], ["Jaden McDaniels", 8.0], ["Dorian Finney-Smith", 8.0], 
    ["Shaedon Sharpe", 8.0], ["Scotty Pippen Jr.", 8.0], ["Bilal Coulibaly", 8.0], ["Malik Beasley", 8.0], 
    ["Dyson Daniels", 7.9], ["Andrew Nembhard", 7.9]]
salary_df = pd.DataFrame(salary_data, columns=['Player', 'Salary'])
#
# Merge the dataset
df = pd.merge(predictions_df, salary_df, on='Player', how='left')
df_filtered = df[df['Salary'].notna() & (df['Salary'] != 0)]
current_players = df_filtered
#
# Creating frontcourt and backcourt distinction
frontcourt_players = current_players[current_players['Pos'] == 1.0] 
backcourt_players = current_players[current_players['Pos'] == 2.0]


# In[ ]:


#
# Create optimization problem
prob = LpProblem("Fantasy_Basketball_Optimization", LpMaximize)
player_vars = {player: LpVariable(f"player_{player}", cat='Binary') for player in current_players['Player']}
prob += lpSum([player_vars[player] * current_players[current_players['Player'] == player]['Total_Score'].values[0] 
               for player in current_players['Player']])
# 
# Add (customizable) constraints...
length_lineup=10
frontcourt=5
backcourt=5
max_salary=100
min_salary=90
per_team=2
#
# 1. Roster length must be exactly 10
prob += lpSum([player_vars[player] for player in current_players['Player']]) == length_lineup
# 2. Frontcourt players must be exactly 5
prob += lpSum([player_vars[player] for player in frontcourt_players['Player']]) == frontcourt
# 3. Backcourt players must be exactly 5
prob += lpSum([player_vars[player] for player in backcourt_players['Player']]) == backcourt
# 4. Max salary constraint
prob += lpSum([player_vars[player] * current_players[current_players['Player'] == player]['Salary'].values[0] 
               for player in current_players['Player']]) <= max_salary
prob += lpSum([player_vars[player] * current_players[current_players['Player'] == player]['Salary'].values[0] 
               for player in current_players['Player']]) >= min_salary
# 5. Only 2 players from the same team
teams = current_players['Tm'].unique()
for team in teams:
    prob += lpSum([player_vars[player] for player in current_players[current_players['Tm'] == team]['Player']]) <= per_team
#
# Solve the optimization problem
prob.solve()
#
# Find the selected players, estimated fantasy points per game, and total salary
selected_players = [player for player in current_players['Player'] if player_vars[player].varValue == 1]
total_score = sum(current_players[current_players['Player'].isin(selected_players)]['Total_Score'])
total_salary = sum(current_players[current_players['Player'].isin(selected_players)]['Salary'])
#
print("Estimated Fantasy Score: ",total_score)
print("Total Salary: ",total_salary)
print("Selected players: ",selected_players)


# In[ ]:


#
# Get the data
players = selected_players
#
# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
#
# Create a title and list names
ax.text(0.5, 1.0, "Our Selected Players:", ha='center', va='center', fontsize=16, fontweight='bold')
for i, player in enumerate(players):
    ax.text(0.5, 0.8 - (i * 0.08), player, ha='center', va='center', fontsize=12)
#
# Save the plot
plt.savefig('selected_players.png', format='png')
plt.close()


# The optimized NBA Fantasy Basketball lineup, based on predicted players' performances using a hypertuned RandomForestRegression model is shown above. In theory, this combination of players will maximize one's chance of winning your NBA Fantasy Basketball league. 
