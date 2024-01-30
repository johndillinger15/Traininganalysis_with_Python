from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Load dash app
app = Dash(__name__)

# Load data from Excel file
df = pd.read_excel("training_data_new.xlsx", sheet_name="2018+", usecols="C:O,S:AD")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Datum'])

# Extract year and month from the Date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
today = pd.to_datetime('today').strftime('%d.%m.%Y')

# Creating Different Dataframes
current_year = pd.to_datetime('today').year
df_ytd = df[(df['Year'] == current_year) & (df['Date'] <= pd.to_datetime('today'))].copy()
df_since_2021 = df[df['Year'] >= 2021]
df_current_year = df[df['Date'].dt.year == current_year]
df_current_year_filtered = df_current_year[['Date', 'KM']].copy()
df_current_year_filtered = df_current_year_filtered.dropna(subset=['KM'])
df_current_year_filtered.loc[:, 'actual'] = df_current_year_filtered['KM'].cumsum()

# Define Running Goal for current year
# Filter data for the last 5 years
last_5_years_df = df[(df['Year'] >= current_year - 5) & (df['Year'] < current_year - 0)]

# Calculate yearly average for 'KM' column
yearly_avg = last_5_years_df.groupby('Year')['KM'].sum()


# Store the average of the last 5 years in running_goal variable
running_goal = int(yearly_avg.mean())

# Create dataframe for running goal
daily_goals_df = pd.DataFrame({'Date': pd.date_range(start=f'{current_year}-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))})
daily_goals_df['Daily_Goal'] = running_goal / 365  # Assuming an equal distribution throughout the year
daily_goals_df['goal'] = daily_goals_df['Daily_Goal'].cumsum()

# Merge the daily goals DataFrame with the current year DataFrame
merged_df_goal = pd.merge(daily_goals_df, df_current_year_filtered, on='Date', how='left')
merged_df_goal = merged_df_goal.dropna(subset=['KM'])

# Calculate the difference between goal and actual
merged_df_goal['goal_difference'] = merged_df_goal['actual'] - merged_df_goal['goal']

# Find the most recent value
todays_goal_difference = merged_df_goal.iloc[-1]['goal_difference']
todays_goal_difference = int(todays_goal_difference)

# Calculate the date 365 days ago from today
end_date_week = datetime.today()
start_date_week = end_date_week - timedelta(days=365)

# Limit the dataframe to the last 365 days
last_365_days = df[(df['Date'] >= start_date_week) & (df['Date'] <= end_date_week)]

# Group by week and sum the values
weekly_data = last_365_days.groupby(pd.Grouper(key='Date', freq='W-Sun')).agg({'KM': 'sum', 'TSS': 'sum'}).reset_index()
print(weekly_data)

# Round the values in 'KM' and 'RSS' to integers with 0 decimals
weekly_data['KM'] = weekly_data['KM'].round(0).astype(int)
weekly_data['TSS'] = weekly_data['TSS'].round(0).astype(int)

# Format the 'Date' column in German time format
df_ytd['Date'] = df_ytd['Date'].dt.strftime('%d.%m.%Y')

# Drop empty strings or NaN values from the 'Schuh' column
cleaned_shoe_data = df['Schuh'].dropna().str.strip()

# Sort the unique shoe values in descending order
sorted_shoe_values = cleaned_shoe_data.unique()[::-1]

# Dropdown options for shoes
shoe_options = [{'label': Schuh, 'value': Schuh} for Schuh in sorted_shoe_values]

# Calculate the sum of Kilometers run with kids
km_sum_kids = df.groupby('k')['KM'].sum().reset_index()
km_sum_kids_j = km_sum_kids[km_sum_kids['k'] == 'j']

# Calculate the sum of kilometers run with kids for YTD
ytd_sum_kids = df_ytd.groupby('k')['KM'].sum().reset_index()
ytd_km_sum_kids_j = ytd_sum_kids[ytd_sum_kids['k'] == 'j']

# Callback function to update the table based on selected shoes
@app.callback(
    Output('running-km-table', 'data'),
    [Input('shoe-dropdown', 'value')],
    [State('shoe-dropdown', 'value')]  # State to remember the selection
)

def update_running_km_table(selected_shoes, remembered_selection):
    if selected_shoes is None:
        if remembered_selection:
            selected_shoes = remembered_selection
        else:
            selected_shoes = df['Schuh'].unique()

    filtered_df = df[df['Schuh'].isin(selected_shoes)]
    running_km_data = filtered_df.groupby('Schuh')['KM'].sum().reset_index()

    # Round the 'KM' column to integer values
    running_km_data['KM'] = running_km_data['KM'].round().astype(int)

    # Sort the running_km_data DataFrame by 'KM' in descending order
    running_km_data = running_km_data.sort_values(by='KM', ascending=False)

    return running_km_data.to_dict('records')

# Set the start date
start_date = pd.to_datetime('2023-01-01')

# Calculate the cumulative sum of 'KM' for each day starting from January 1, 2024, for the trailing 90 days
cumulative_sum_90_days = []
cumulative_sum_365_days = []

for date in pd.date_range(start_date, pd.to_datetime('today')):
    trailing_90_days_sum = df[(df['Date'] >= (date - pd.DateOffset(days=89))) & (df['Date'] <= date)]['KM'].sum()
    cumulative_sum_90_days.append(trailing_90_days_sum)

for date in pd.date_range(start_date, pd.to_datetime('today')):
    trailing_365_days_sum = df[(df['Date'] >= (date - pd.DateOffset(days=364))) & (df['Date'] <= date)]['KM'].sum()
    cumulative_sum_365_days.append(trailing_365_days_sum)

# Create a DataFrame for the results
result_df_90_days = pd.DataFrame({'Date': pd.date_range(start_date, pd.to_datetime('today')), 'Cumulative_Sum_90_Days': cumulative_sum_90_days})
result_df_365_days = pd.DataFrame({'Date': pd.date_range(start_date, pd.to_datetime('today')), 'Cumulative_Sum_365_Days': cumulative_sum_365_days})

merged_df = pd.merge(result_df_90_days, result_df_365_days, on='Date', how='outer')

# PWR/HR

# Calculate the cumulative average of 'pwr/hr' for each day starting from January 1, 2024, for the trailing 42 days
cumulative_avg_pwr_hr_42_days = []

for date in pd.date_range(start_date, pd.to_datetime('today')):
    trailing_42_days_avg_pwr_hr = df[(df['Date'] >= (date - pd.DateOffset(days=41))) & (df['Date'] <= date)]['pwr/hr'].mean()
    cumulative_avg_pwr_hr_42_days.append(trailing_42_days_avg_pwr_hr)

# Create a DataFrame for the results
result_df_avg_pwr_hr_42_days = pd.DataFrame({'Date': pd.date_range(start_date, pd.to_datetime('today')), 'Cumulative_Avg_pwr_hr_42_Days': cumulative_avg_pwr_hr_42_days})


# Changing decimals of numbers in dataframes

# Rounding the last value in the 'CP' column to 0 decimal places
rounded_CP_value = int(round(df_ytd['CP'].iloc[-1], 0))

# Rounding the last value in the 'RSS load' column to 2 decimal places
rounded_load_value= round(df_ytd['RSS load'].iloc[-1], 2)

# Rounding the last value in the 'rCTL' column to 1 decimal place
rounded_rCTL_value= round(df_ytd['rCTL'].iloc[-1], 1)

# Rounding the trailing 90d km value to 0 decimal place
YTD_km_data = int(round(df_ytd['KM'].sum()))

# Rounding trailing 42d avg of Pwr/HR to 2 decimal places
rounded_pwr_hr_42 = round(result_df_avg_pwr_hr_42_days['Cumulative_Avg_pwr_hr_42_Days'].iloc[-1],2)

# YTD Höhenmeter
YTD_HM_data = int(round(df_ytd['HM'].sum()))

# Rounding total sum of KM with kids
rounded_sum_kids = int(round(km_sum_kids_j['KM'].iloc[-1],0))
rounded_ytd_km_sum_kids_j = int(round(ytd_km_sum_kids_j['KM'].iloc[-1],0))

# Calculate running times

# Distances in meters
distance_5k = 5000
distance_10k = 10000
distance_HM = 21097.5  # Half Marathon
distance_marathon = 42195  # Marathon

# % of CP for each distance
modification_factor_5k = 1.06
modification_factor_10k = 1.01
modification_factor_HM = 0.96
modification_factor_marathon = 0.9

# target race CP
CP_5k = modification_factor_5k * rounded_CP_value
CP_5k = int(CP_5k)
CP_10k = modification_factor_10k * rounded_CP_value
CP_10k = int(CP_10k)
CP_HM = modification_factor_HM * rounded_CP_value
CP_HM = int(CP_HM)
CP_M = modification_factor_marathon * rounded_CP_value
CP_M = int(CP_M)

# weight
m = 67

def calculate_time(distance, CP, m, modification_factor):
    # Apply modification factor
    modified_CP = CP * modification_factor
    
    # Calculate speed using the simplified formula v = P/m/1.04
    speed = modified_CP / m / 1.04
    
    # Calculate time T = 1.04 * d / (P/m)
    time_seconds = 1.04 * distance / (modified_CP / m)
    
    # Convert time to hh:mm:ss format
    time_delta = timedelta(seconds=time_seconds)
    time_formatted = str(time_delta).split(".")[0]  # Remove milliseconds
    
    return time_formatted

# Calculate times for each distance
time_5k = calculate_time(distance_5k, rounded_CP_value, m, modification_factor_5k)
time_10k = calculate_time(distance_10k, rounded_CP_value, m, modification_factor_10k)
time_HM = calculate_time(distance_HM, rounded_CP_value, m, modification_factor_HM)
time_marathon = calculate_time(distance_marathon, rounded_CP_value, m, modification_factor_marathon)


# Making the figures

# Plotting the first line chart (tCTL vs rCTL) using Plotly Express
fig1 = px.line(df_ytd, x='Date', y=['tCTL', 'rCTL'], title=f'YTD Comparison of CTL',
              labels={'value': 'CTL', 'Date': 'Date', 'variable': 'CTL'},
              line_shape='linear', 
              render_mode='svg')

fig1.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig1.add_vline(x="27.01.2024", line_width=3, line_dash="dash", line_color="green")

# Plotting the second line chart (TSS vs RSS) using Plotly Express

# Create the line chart without background coloring
fig2 = px.line(df_ytd, x='Date', y=['TSS load', 'RSS load'], title=f'YTD Comparison of load for {current_year}',
              labels={'value': 'load', 'variable': 'load'},
              line_shape='linear', render_mode='svg')

fig2.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig2.add_vline(x="27.01.2024", line_width=3, line_dash="dash", line_color="green")

# Define the shading ranges
shading_ranges = [
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 0.5, 'y1': 0.8, 'color': 'rgba(53, 77, 115,0.5)'},  # blue
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 0.8, 'y1': 1, 'color': 'rgba(0, 200, 0,0.5)'},  # lightgreen
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1, 'y1': 1.3, 'color': 'rgba(0,255,100,0.5)'},  # darkgreen
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1.3, 'y1': 1.5, 'color' : 'rgba(255,255,0,0.5)'},  # Yellow
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1.5, 'y1': df_ytd[['TSS load', 'RSS load']].max().max(), 'color': 'rgba(255,0,0,0.5)'}  # Red
]

# Add the shaded regions to the plot
for shading_range in shading_ranges:
    fig2.add_shape(type="rect",
                  x0=shading_range['x0'], x1=shading_range['x1'],
                  y0=shading_range['y0'], y1=shading_range['y1'],
                  fillcolor=shading_range['color'], opacity=0.5, layer='below', line_width=0)

# Plotting the third line chart FTP vs CP using Plotly Express
fig3 = px.line(df_ytd, x='Date', y=['FTP', 'CP'], title=f'YTD Comparison of CP vs FTP for {current_year}',
              labels={'value': 'CP/FTP', 'variable': 'Metric'},
              line_shape='linear', render_mode='svg')

fig3.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig3.add_vline(x="27.01.2024", line_width=3, line_dash="dash", line_color="green")

# Plotting daily RSS vs TSS
fig4 = px.bar(df_ytd, x='Date', y=['TSS', 'RSS'], title=f'TSS vs RSS per Run', barmode='group', text='Art')

fig4.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig4.add_vline(x="27.01.2024", line_width=3, line_dash="dash", line_color="green")

# Monthly running progress
fig5 = px.histogram(df, x='Year', y='KM', color='Year', title='Yearly running progress')

fig5.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    showlegend=False,
    bargap=0.2
)

# Monthly elevation gain

# Create the histogram
fig6 = px.histogram(df_since_2021, x='Month', y='HM', color='Year', barmode='group', title='Höhenmeter pro Monat')

fig6.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    showlegend=False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)

# Create line chart for the running volume of last 90 and 365 days

# Create a figure with two subplots
fig8 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for Cumulative_Sum_90_Days on the left y-axis
fig8.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_90_Days'], mode='lines', name='Trailing 90days km'))

# Add traces for Cumulative_Sum_365_Days on the right y-axis
fig8.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_365_Days'], mode='lines', name='Trailing 365days km'), secondary_y=True)

# Update layout with titles and labels
fig8.update_layout(title='Trailing 90d and 365d running volume',
                  xaxis_title='Date',
                  yaxis_title='Trailing 90 Days',
                  yaxis2_title='Trailing 365 Days',
                  legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
                  legend_title=None,
                  width=720,  # Set the width of the graph
                  height=390,  # Set the height of the graph
                  plot_bgcolor="white",
)

# PWR/HR Graph
fig9 = px.line(result_df_avg_pwr_hr_42_days, x='Date', y='Cumulative_Avg_pwr_hr_42_Days', title='Avg of pwr/hr for Trailing 42 Days',
              labels={'Cumulative_Avg_pwr_hr_90_Days': 'Avg pwr/hr', 'Date': 'Date'})

fig9.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# Weekly Data
# Create a Plotly figure
fig11 = px.bar(weekly_data, x='Date', y='KM', labels={'KM': 'KM'}, title='Weekly KM vs RSS')

# Add a bar trace for 'RSS' on the secondary y-axis
fig11.add_trace(px.line(weekly_data, x='Date', y='TSS', labels={'TSS': 'TSS'}, color_discrete_sequence=['firebrick']).update_traces(yaxis='y2').data[0])

fig11.update_layout(
    yaxis2=dict(title='TSS', overlaying='y', side='right'),
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white"
)

# Yearly running goal graph
fig13 = px.line(merged_df_goal, x='Date', y=['actual', 'goal'], title=f'Yearly Progress in {current_year} vs Goal (KM)')

fig13.update_layout(
    width=720,  # Set the width of the graph
    height=390,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# DASHBOARD

app.layout = html.Div(children=[
html.H2(children=f"Dashboard Running Data for {today}"),
    html.Div(children=[    
        html.P(f"Heutige CP: {rounded_CP_value} W"),
        html.P(f"Heutige load: {rounded_load_value}"),
        html.P(f"Heutige CTL: {rounded_rCTL_value}"),
        html.P(f"Pwr/Hr (Vergangene 42 Tage): {rounded_pwr_hr_42}"),
    ], style={'display': 'inline-block', 'width': '25%'}),  # Adjust width as needed

    html.Div(children=[    
        html.P(f"Laufvolumen YTD: {YTD_km_data} km"),
        html.P(f"{current_year} goal: {running_goal} km"),
        html.P(f"Unterschied zum {current_year} Ziel: {todays_goal_difference} km"),
    ], style={'display': 'inline-block', 'width': '25%'}),  # Adjust width as needed

        html.Div(children=[    
        html.P(f"Höhenmeter YTD: {YTD_HM_data} m"),
        html.P(f"Kilometer mit Kids: {rounded_sum_kids} km"),
        html.P(f"Kilometer mit Kids {current_year}: {rounded_ytd_km_sum_kids_j} km"),
    ], style={'display': 'inline-block', 'width': '25%'}),  # Adjust width as needed

    html.Div(children=[    
        html.P(f"5k: {time_5k} ({CP_5k} W)"),
        html.P(f"10k: {time_10k} ({CP_10k} W)"),
        html.P(f"Half Marathon: {time_HM} ({CP_HM} W)"),
        html.P(f"Marathon: {time_marathon} ({CP_M} W)"),
    ], style={'display': 'inline-block', 'width': '25%'}),  # Adjust width as needed   

    
html.H2(children='Power Values'),

html.H3(children="RSS vs TSS Vergleich"),

# CTL and load
html.Div([
        # CTL copmarison
        html.Div(dcc.Graph(figure=fig1)),
        # load comparison
        html.Div(dcc.Graph(figure=fig2)),
    ], style={'display': 'flex'}),

# FTP vs CP and daily RSS vs TSS
html.Div([
        # third figure
        html.Div(dcc.Graph(figure=fig3)),
        # forth figure
        html.Div(dcc.Graph(figure=fig4)),
    ], style={'display': 'flex'}),

# PWR/HR
html.Div([
        # pwr/hr
        html.Div(dcc.Graph(figure=fig9)),
        # Trailing 90 and 365 day volume
        html.Div(dcc.Graph(figure=fig8)),
    ], style={'display': 'flex'}),

html.H3(children="Schuhe"),

# Dropdown for selecting shoes
dcc.Dropdown(
        id='shoe-dropdown',
        options=shoe_options,
        multi=True,
        value=['Adidas Solarglide 5','Adizero Boston 8','HOKA Clifton 8','HOKA Rincon 3 II','Innov-8 Terraultra','Innov-8 Trailfly','Saucony Triumph 21','HOKA Mach 5'],  # Set the default value here
        style={'margin-bottom': '20px', 'width':'70%'}
),

# Table for running shoes
dash_table.DataTable(
        id='running-km-table',
        columns=[
        {'name': 'Schuh', 'id': 'Schuh'},
        {'name': 'Total Running KM', 'id': 'KM'},
    ],
    style_table={
        'maxHeight': '300px',
        'overflowY': 'scroll',
        'width': '30%',
        'margin-top': '20px',
    },
    style_cell={
        'textAlign': 'left',
        'minWidth': '50px', 'maxWidth': '100px',
        'whiteSpace': 'normal',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
    },
    style_data_conditional=[
        {
            'if': {'column_id': 'KM', 'filter_query': '{KM} > 750'},
            'backgroundColor': 'red',
            'color': 'white',
        },
        {
            'if': {'column_id': 'KM', 'filter_query': '{KM} > 500 and {KM} <= 750'},
            'backgroundColor': 'yellow',
        },
    ],),

# Monthly Kilometers
html.H3(children="Monthly and yearly running progress"),
html.Div([
        # Yearly KM
        html.Div(dcc.Graph(figure=fig5)),
    ], style={'display': 'flex'}),

# Höhenmeters
html.H3(children="Elevation gain per month"),
html.Div([
        # weekly running graph
        html.Div(dcc.Graph(figure=fig6)),
    ], style={'display': 'flex'}),     

# Weekly Kilometers
html.H3(children="Weekly running progress and Running Stress Score"),
html.Div([
        # weekly running graph
        html.Div(dcc.Graph(figure=fig11)), 
        # yearly goal graph
        html.Div(dcc.Graph(figure=fig13)), 
    ], style={'display': 'flex'}),
])    

if __name__ == '__main__':
    app.run(debug=True)