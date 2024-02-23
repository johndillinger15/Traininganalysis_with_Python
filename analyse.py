from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Color-scheme
# rgb(178, 34, 34), rgb(153, 97, 0), rgb(113, 135, 38), rgb(64, 162, 111), rgb(34, 180, 180)
# firebrick: rgb(178, 34, 34), complementary blue: rgb(34, 180, 180)
# https://colordesigner.io/gradient-generator/?mode=lch#B22222-22B4B4

# Load dash app
app = Dash(__name__)

# Load data from Excel file
df = pd.read_excel("training_data_new.xlsx", sheet_name="2018+", usecols="C:AG")

# Convert Date and Pace column to datetime and Time and format
df['Date'] = pd.to_datetime(df['Datum'])

# Extract year and month from the Date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
today = pd.to_datetime('today').strftime('%Y-%m-%d')
today2 = pd.to_datetime('today')
tomorrow = datetime.now() + timedelta(days=1)
tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

# Creating Different Dataframes
current_year = pd.to_datetime('today').year
df_ytd = df[(df['Year'] == current_year) & (df['Date'] <= pd.to_datetime('today'))]

df_ytd = df_ytd.copy()
df_ytd.fillna({'KM':0, 'HFQ':0, 'W1':0, 'W2':0, 'HM':0, 'Pace':0}, inplace=True)
df_ytt = df[(df['Date'] >= pd.Timestamp(datetime(tomorrow.year, 1, 1))) & (df['Date'] <= tomorrow)]
df_ytt = df_ytt.copy()  # Explicitly copy the subset of the DataFrame
df_ytt.fillna({'KM':0, 'W1':0, 'W2':0}, inplace=True)
df_since_2020 = df[df['Year'] >= 2020]
df_until_today = df[df['Date'] <= today]
df_current_year = df[df['Date'].dt.year == current_year]
df_current_year_filtered = df_current_year[['Date', 'KM']].copy()
df_current_year_filtered.fillna({'KM':0}, inplace=True)
df_current_year_filtered.loc[:, 'actual'] = df_current_year_filtered['KM'].cumsum()
# Calculate the date 90 days ago
ninety_days_ago = today2 - timedelta(days=90)
# Filter the dataframe for the last 90 days and excluding future dates
df_last_90_days = df[(df['Date'] >= ninety_days_ago) & (df['Date'] <= today2)]

# Define Running Goal for current year
# Filter data for the last 5 years
last_5_years_df = df[(df['Year'] >= current_year - 5) & (df['Year'] < current_year - 0)]

# Calculate yearly average for 'KM' column
yearly_avg = last_5_years_df.groupby('Year')['KM'].sum()

# Store the average of the last 5 years in running_goal variable
running_goal = int(yearly_avg.mean())

# Create dataframe for running goal YTD
daily_goals_df = pd.DataFrame({'Date': pd.date_range(start=f'{current_year}-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))})
daily_goals_df['Daily_Goal'] = running_goal / 365  # Assuming an equal distribution throughout the year
daily_goals_df['goal'] = daily_goals_df['Daily_Goal'].cumsum()

# Create dataframe for running goal whole current year
daily_goals_df_complete = pd.DataFrame({'Date': pd.date_range(start=f'{current_year}-01-01', end=f'{current_year}-12-31')})
daily_goals_df_complete['Daily_Goal'] = running_goal / 365
daily_goals_df_complete['goal'] = daily_goals_df_complete['Daily_Goal'].cumsum()

# Merge the daily goals DataFrame with the current year DataFrame
merged_df_goal = pd.merge(daily_goals_df, df_current_year_filtered, on='Date', how='left')
merged_df_goal_complete = pd.merge(daily_goals_df_complete, df_current_year_filtered, on='Date', how='left')
merged_df_goal.fillna(0, inplace=True)
merged_df_goal_complete.fillna(0, inplace=True)

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

# Group by week and sum the values (last 365d)
weekly_data = last_365_days.groupby(pd.Grouper(key='Date', freq='W-Sun')).agg({'KM': 'sum', 'RSS': 'sum'}).reset_index()

# Round the values in 'KM' and 'RSS' to integers with 0 decimals
weekly_data['KM'] = weekly_data['KM'].round(0).astype(int)
weekly_data['RSS'] = weekly_data['RSS'].round(0).astype(int)

# Group by week and sum the values (current year)
weekly_data_current_year = df_current_year.groupby(pd.Grouper(key='Date', freq='W-Sun')).agg({'KM': 'sum', 'RSS': 'sum'}).reset_index()

# Round the values in 'KM' and 'RSS' to integers with 0 decimals
weekly_data_current_year['KM'] = weekly_data_current_year['KM'].round(0).astype(int)
weekly_data_current_year['RSS'] = weekly_data_current_year['RSS'].round(0).astype(int)

#Schuhdaten
df_Schuhe = df.groupby('Schuh')['KM'].sum().reset_index()
active_shoes = ['Adidas Solarglide 5','Adizero Boston 8','HOKA Clifton 8','HOKA Rincon 3 II','Innov-8 Terraultra','Innov-8 Trailfly','Saucony Triumph 21','HOKA Mach 5']
df_Schuhe = df_Schuhe[df_Schuhe['Schuh'].isin(active_shoes)]
df_Schuhe = df_Schuhe.sort_values(by='KM', ascending=False)
df_Schuhe = df_Schuhe.to_dict('records')

# Calculate the sum of Kilometers run with kids
km_sum_kids = df.groupby('k')['KM'].sum().reset_index()
km_sum_kids_j = km_sum_kids[km_sum_kids['k'] == 'j']

# Calculate the sum of kilometers run with kids for YTD
ytd_sum_kids = df_ytd.groupby('k')['KM'].sum().reset_index()
ytd_km_sum_kids_j = ytd_sum_kids[ytd_sum_kids['k'] == 'j']

# Set the start date
start_date = end_date_week - timedelta(days=365)

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

# Today's workout
todays_workout = df_ytd['Art'].iloc[-1]
todays_km = df_ytd['KM'].iloc[-1]
todays_time = df_ytd['Zeit'].iloc[-1]
todays_pace = df_ytd['Pace'].iloc[-1]
todays_watt = df_ytd['Watt'].iloc[-1]
todays_HFQ = df_ytd['HFQ'].iloc[-1]
todays_HFQ = int(round(todays_HFQ, 0))
todays_HM = df_ytd['HM'].iloc[-1]
todays_HM = int(round(todays_HM, 0))
todays_ATL = df_ytd['rATL'].iloc[-1]
todays_ATL = round(todays_ATL, 1)
todays_load = df_ytd['RSS load'].iloc[-1]
todays_load = round(todays_load, 2)
todays_zone_start = df_ytd['W1'].iloc[-1]
todays_zone_start = int(round(todays_zone_start, 0))
todays_zone_end = df_ytd['W2'].iloc[-1]
todays_zone_end = int(round(todays_zone_end, 0))
tomorrows_workout = df_ytt['Art'].iloc[-1]
tomorrows_km = df_ytt['KM'].iloc[-1]
tomorrows_km = round(tomorrows_km, 2)
tomorrows_time = df_ytt['Zeit'].iloc[-1]
tomorrows_zone_start = df_ytt['W1'].iloc[-1]
tomorrows_zone_start = int(round(tomorrows_zone_start, 0))
tomorrows_zone_end = df_ytt['W2'].iloc[-1]
tomorrows_zone_end = int(round(tomorrows_zone_end, 0))
max_CP = df_ytd['CP'].max()
max_CP = int(round(max_CP, 0))

# handling today's Pace
if todays_pace != 0:
    # Extracting minutes and seconds
    minutes = todays_pace.minute
    seconds = todays_pace.second

    # Formatting as mm:ss
    formatted_todays_pace = "{:02d}:{:02d}".format(minutes, seconds)
else:
    formatted_todays_pace = "0"

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
modification_factor_5k = 1.065
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

# ECOR
ECOR_5k = 1.049
ECOR_10k = 1.060
ECOR_HM = 1.060
ECOR_M = 1.067

def calculate_time(ECOR, distance, CP, m, modification_factor):
    # Apply modification factor
    modified_CP = CP * modification_factor
    
    # Calculate speed using the simplified formula v = P/m/1.04
    speed = modified_CP / m / ECOR
    
    # Calculate time T = 1.04 * d / (P/m)
    time_seconds = ECOR * distance / (modified_CP / m)
    
    # Convert time to hh:mm:ss format
    time_delta = timedelta(seconds=time_seconds)
    time_formatted = str(time_delta).split(".")[0]  # Remove milliseconds
    
    return time_formatted

# Calculate times for each distance
time_5k = calculate_time(ECOR_5k, distance_5k, rounded_CP_value, m, modification_factor_5k)
time_10k = calculate_time(ECOR_10k, distance_10k, rounded_CP_value, m, modification_factor_10k)
time_HM = calculate_time(ECOR_HM, distance_HM, rounded_CP_value, m, modification_factor_HM)
time_marathon = calculate_time(ECOR_M, distance_marathon, rounded_CP_value, m, modification_factor_marathon)


# Making the figures
# Plotting the first line chart (tCTL vs rCTL) using Plotly Express
fig1 = px.line(df_ytd, x='Date', y=['tCTL', 'rCTL'], title=f'tCTL vs rCTL (fig1)',
               labels={'value': 'CTL'}, color_discrete_sequence=['#2283B4','firebrick'])

fig1.update_layout(
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig1.add_vline(x="2024-01-27", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")

# RSS per run and CTL
fig1neu = px.bar(df_last_90_days, x='Date', y=['RSS'], barmode='group', text='Art',
              labels={'value':'RSS'}, color_discrete_sequence=['#2283B4','firebrick'])

fig1neu.add_trace(go.Scatter(x=df_last_90_days['Date'], y=df_last_90_days['rCTL'], mode='lines', name='CTL', line=dict(color='firebrick')))

fig1neu.update_layout(
    title='RSS and CTL for last 90d (fig1neu)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# YTD load comparison
fig2 = px.line(df_ytd, x='Date', y=['TSS load', 'RSS load'], title=f'YTD load comparison of {current_year} (fig2)',
              labels={'value': 'load', 'variable': 'load'},
              line_shape='linear', color_discrete_sequence=['#2283B4','firebrick'], range_y=[0.5,2])

fig2.update_layout(
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=False)

fig2.add_vline(x="2024-01-27", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")

# Define the shading ranges
shading_ranges = [
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 0.5, 'y1': 0.8, 'color': 'rgba(53, 77, 115,0.5)'},  # blue
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 0.8, 'y1': 1, 'color': 'rgba(0, 200, 0,0.5)'},  # lightgreen
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1, 'y1': 1.3, 'color': 'rgba(0,255,100,0.5)'},  # darkgreen
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1.3, 'y1': 1.5, 'color' : 'rgba(255,255,0,0.5)'},  # Yellow
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1.5, 'y1': 2, 'color': 'rgba(255,0,0,0.5)'}  # Red
]

# Add the shaded regions to the plot
for shading_range in shading_ranges:
    fig2.add_shape(type="rect",
                  x0=shading_range['x0'], x1=shading_range['x1'],
                  y0=shading_range['y0'], y1=shading_range['y1'],
                  fillcolor=shading_range['color'], opacity=0.5, layer='below', line_width=0)
    
# Load for last 90d
fig2neu = px.line(df_last_90_days, x='Date', y=['RSS load'],
              labels={'value': 'load', 'variable': 'load'},
              line_shape='linear', color_discrete_sequence=['#2283B4','firebrick'], range_y=[0.5,2])

fig2neu.update_layout(
    title='load for last 90d (fig2neu)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)
fig2neu.update_xaxes(showgrid=False)
fig2neu.update_yaxes(showgrid=False)

# Define the shading ranges
shading_ranges_new = [
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 0.5, 'y1': 0.8, 'color': 'rgba(53, 77, 115,0.5)'},  # blue
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 0.8, 'y1': 1, 'color': 'rgba(0, 200, 0,0.5)'},  # lightgreen
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 1, 'y1': 1.3, 'color': 'rgba(0,255,100,0.5)'},  # darkgreen
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 1.3, 'y1': 1.5, 'color' : 'rgba(255,255,0,0.5)'},  # Yellow
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 1.5, 'y1': 2, 'color': 'rgba(255,0,0,0.5)'}  # Red
]
for shading_range in shading_ranges_new:
    fig2neu.add_shape(type="rect",
                  x0=shading_range['x0'], x1=shading_range['x1'],
                  y0=shading_range['y0'], y1=shading_range['y1'],
                  fillcolor=shading_range['color'], opacity=0.5, layer='below', line_width=0)


# Plotting the third line chart FTP vs CP using Plotly Express
fig3 = px.line(df_ytd, x='Date', y=['FTP', 'CP'], title=f'YTD comparison: CP vs FTP ({current_year}) (fig3)',
              labels={'value': 'CP/FTP', 'variable': 'Metric'},
              line_shape='linear', color_discrete_sequence=['#2283B4','firebrick'])

fig3.update_layout(
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig3.add_vline(x="2024-01-27", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")

# Plotting daily RSS vs TSS
fig4 = px.bar(df_ytd, x='Date', y=['TSS', 'RSS'], title=f'TSS vs RSS per Run (fig4)', barmode='group', text='Art',
              labels={'value':'TSS/RSS'}, color_discrete_sequence=['#2283B4','firebrick'])

fig4.update_layout(
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig4.add_vline(x="2024-01-27", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")

# Monthly running progress
fig5 = px.histogram(df_until_today, x='Year', y='KM', color='Year', color_discrete_sequence=['rgb(34, 180, 180)'])

fig5.update_layout(
    title='Yearly Running Volume (fig5)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
    bargap=0.2,
    plot_bgcolor="white",    
    yaxis_title='KM',
)

# Monthly elevation gain

# Define the order of months
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the histogram
fig6 = px.histogram(df_since_2020, x='Month', y='HM', color='Year', barmode='group', category_orders={'Month': month_order}, color_discrete_sequence=['rgb(178, 34, 34)', 'rgb(153, 97, 0)', 'rgb(113, 135, 38)', 'rgb(64, 162, 111)', 'rgb(34, 180, 180)'])

fig6.update_layout(
    title='Elevation Gain Per Month (fig6)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1, # gap between bars of the same location coordinates
    plot_bgcolor="white",    
    yaxis_title='HM',
)

# Update x-axis tick labels
fig6.update_xaxes(tickmode='array', tickvals=list(range(1, 13)), ticktext=month_order)

# Create line chart for the running volume of last 90 and 365 days
# Create a figure with two subplots
fig8 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for Cumulative_Sum_90_Days on the left y-axis
fig8.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_90_Days'], mode='lines', name='Trailing 90days km', line=dict(color='firebrick')),)

# Add traces for Cumulative_Sum_365_Days on the right y-axis
fig8.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_365_Days'], mode='lines', name='Trailing 365days km', line=dict(color='rgb(34, 180, 180)')), secondary_y=True)

# Update layout with titles and labels
fig8.update_layout(title='Trailing 90d and 365d Running Volume (fig8)',
                  xaxis_title='Date',
                  yaxis_title='Trailing 90 Days',
                  yaxis2_title='Trailing 365 Days',
                  legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
                  legend_title=None,
                  width=640,  # Set the width of the graph
                  height=400,  # Set the height of the graph
                  plot_bgcolor="white",
)

# add shaded region
fig8.add_vrect(x0="2023-10-28", x1="2023-12-28", fillcolor="firebrick", line_width=0, opacity=0.2)

# PWR/HR Graph
fig9 = px.line(result_df_avg_pwr_hr_42_days, x='Date', y='Cumulative_Avg_pwr_hr_42_Days', title='Avg of pwr/hr for Trailing 42 Days (fig9)',
              labels={'Cumulative_Avg_pwr_hr_90_Days': 'Avg pwr/hr'}, color_discrete_sequence=['#2283B4'])

fig9.update_layout(
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    yaxis_title='Pwr/Hr',
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# add shaded region
fig9.add_vrect(x0="2023-10-28", x1="2023-12-28", fillcolor="firebrick", line_width=0, opacity=0.2)

# Weekly Data last 365d
# Create a Plotly figure
fig11 = px.bar(weekly_data, x='Date', y='KM', labels={'KM': 'KM'}, color_discrete_sequence=['rgb(34, 180, 180)'])

# Add a bar trace for 'RSS' on the secondary y-axis
fig11.add_trace(px.line(weekly_data, x='Date', y='RSS', labels={'RSS': 'RSS'}, color_discrete_sequence=['firebrick']).update_traces(yaxis='y2').data[0])

fig11.update_layout(
    title='Weekly KM vs RSS (fig11)',
    yaxis2=dict(title='RSS', overlaying='y', side='right', range=[0, weekly_data['RSS'].max()]),
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white"
)

# Yearly YTD running goal graph
fig13 = px.line(merged_df_goal, x='Date', y=['actual', 'goal'],  labels={'value':'KM'}, color_discrete_sequence=['rgb(34, 180, 180)','firebrick'])

fig13.update_layout(
    title=f'YTD Progress in {current_year} vs Goal (KM) (fig13)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# 2024 monthly kilometer
fig14 = px.histogram(df_current_year, x='Date', y='KM', barmode='group', color_discrete_sequence=['rgb(34, 180, 180)'])

fig14.update_layout(
    title=f'Monthly Volumes for {current_year} (fig14)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1, # gap between bars of the same location coordinates
    plot_bgcolor="white",    
    yaxis_title='KM',
)
fig14.update_xaxes(ticklabelmode='period')

# Yearly YTD running goal graph
fig15 = go.Figure()
fig15.add_trace(go.Scatter(x=merged_df_goal_complete['Date'],y=merged_df_goal_complete['goal'],
    fill=None,
    mode='lines',
    line_color='firebrick',
    ))
fig15.add_trace(go.Scatter(x=merged_df_goal_complete['Date'],y=merged_df_goal_complete['actual'],
    fill='tonexty', # fill area between trace0 and trace1
    mode='lines', line_color='rgb(34,180,180)', fillcolor='rgba(64, 162, 111, 0.3)'),)
fig15.update_layout(
    title=f'Goal vs actual {current_year} (fig15)',
    width=1300,  # Set the width of the graph
    height=750,  # Set the height of the graph
    showlegend=False,
    plot_bgcolor="white",
)
fig15.add_vline(x=f"{today}", line_width=2, line_color="rgb(153, 97, 0)")


# elevation gain per year
fig16 = px.histogram(df_since_2020, x='Year', y='HM', color='Year', color_discrete_sequence=['rgb(178, 34, 34)', 'rgb(153, 97, 0)', 'rgb(113, 135, 38)', 'rgb(64, 162, 111)', 'rgb(34, 180, 180)'])

fig16.update_layout(
    title='Elevation Gain Per Year (fig16)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1, # gap between bars of the same location coordinates
    plot_bgcolor="white",    
    yaxis_title='HM',
)

fig16.add_hline(y=10000, line_width=1, line_dash='dash', line_color="firebrick")

# Weekly Data current year
# Create a Plotly figure
fig17 = px.bar(weekly_data_current_year, x='Date', y='KM', labels={'KM': 'KM'}, color_discrete_sequence=['rgb(34, 180, 180)'])

# Add a bar trace for 'RSS' on the secondary y-axis
fig17.add_trace(px.line(weekly_data_current_year, x='Date', y='RSS', labels={'RSS': 'RSS'}, color_discrete_sequence=['firebrick']).update_traces(yaxis='y2').data[0])

fig17.update_layout(
    title=f'Weekly KM vs RSS for {current_year} (fig17)',
    yaxis2=dict(title='TSS', overlaying='y', side='right', range=[0, weekly_data_current_year['RSS'].max()]),
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white"
)
fig17.add_vline(x=f"{today}", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")

# This year's load
fig18 = px.line(df_current_year, x='Date', y=['TSS load', 'RSS load'],
              labels={'value': 'load', 'variable': 'load'},
              line_shape='linear', color_discrete_sequence=['rgb(34, 180, 180)','firebrick'], range_y=[0.5,2])

fig18.update_layout(
    title=f'Load comparison of {current_year} (fig18)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)
fig18.update_xaxes(showgrid=False)
fig18.update_yaxes(showgrid=False)

fig18.add_vline(x=f"{today}", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")


# Define the shading ranges
shading_ranges_current = [
    {'x0': df_current_year['Date'].min(), 'x1': df_current_year['Date'].max(), 'y0': 0.5, 'y1': 0.8, 'color': 'rgba(53, 77, 115,0.5)'},  # blue
    {'x0': df_current_year['Date'].min(), 'x1': df_current_year['Date'].max(), 'y0': 0.8, 'y1': 1, 'color': 'rgba(0, 200, 0,0.5)'},  # lightgreen
    {'x0': df_current_year['Date'].min(), 'x1': df_current_year['Date'].max(), 'y0': 1, 'y1': 1.3, 'color': 'rgba(0,255,100,0.5)'},  # darkgreen
    {'x0': df_current_year['Date'].min(), 'x1': df_current_year['Date'].max(), 'y0': 1.3, 'y1': 1.5, 'color' : 'rgba(255,255,0,0.5)'},  # Yellow
    {'x0': df_current_year['Date'].min(), 'x1': df_current_year['Date'].max(), 'y0': 1.5, 'y1': 2, 'color': 'rgba(255,0,0,0.5)'}  # Red
]

# Add the shaded regions to the plot
for shading_range_current in shading_ranges_current:
    fig18.add_shape(type="rect",
                  x0=shading_range_current['x0'], x1=shading_range_current['x1'],
                  y0=shading_range_current['y0'], y1=shading_range_current['y1'],
                  fillcolor=shading_range_current['color'], opacity=0.5, layer='below', line_width=0)


# Violin Plot for RSS/TSS

fig19 = px.box(df_ytd, y='RSS/TSS', points='all', color='Art', color_discrete_sequence=['rgb(178, 34, 34)', 'rgb(153, 97, 0)', 'rgb(113, 135, 38)'])

fig19.update_layout(
    title='RSS/TSS (fig19)',
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# Fig20 CTL vs 42d average of RSS
fig20 = px.line(df_ytd, x='Date', y=['rCTL', '42d avg'], title=f'rCTL vs 42d avg (fig20)',
               labels={'value': 'CTL'}, color_discrete_sequence=['#2283B4','firebrick'])

fig20.update_layout(
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig20.add_vline(x="2024-01-27", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")

# YTD load comparison RSS vs 42d avg
fig21 = px.line(df_ytd, x='Date', y=['TSS load', 'load (avg)'], title=f'YTD load comparison of {current_year} (fig21)',
              labels={'value': 'load', 'variable': 'load'},
              line_shape='linear', color_discrete_sequence=['#2283B4','firebrick'])

fig21.update_layout(
    width=640,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

fig21.update_xaxes(showgrid=False)
fig21.update_yaxes(showgrid=False)

fig21.add_vline(x="2024-01-27", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")

# Define the shading ranges
shading_ranges = [
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 0.5, 'y1': 0.8, 'color': 'rgba(53, 77, 115,0.5)'},  # blue
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 0.8, 'y1': 1, 'color': 'rgba(0, 200, 0,0.5)'},  # lightgreen
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1, 'y1': 1.3, 'color': 'rgba(0,255,100,0.5)'},  # darkgreen
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1.3, 'y1': 1.5, 'color' : 'rgba(255,255,0,0.5)'},  # Yellow
    {'x0': df_ytd['Date'].min(), 'x1': df_ytd['Date'].max(), 'y0': 1.5, 'y1': 2, 'color': 'rgba(255,0,0,0.5)'}  # Red
]

# Add the shaded regions to the plot
for shading_range in shading_ranges:
    fig21.add_shape(type="rect",
                  x0=shading_range['x0'], x1=shading_range['x1'],
                  y0=shading_range['y0'], y1=shading_range['y1'],
                  fillcolor=shading_range['color'], opacity=0.5, layer='below', line_width=0)

# DASHBOARD

app.layout = html.Div([
    dcc.Tabs([
# Tab 1 - Wertetabelle        
dcc.Tab(label='Important Values', children=[
    
#   html.H2(f"Today's Workout ({today})"),
    html.Div(
    children=[    
        html.P(children=[
            html.Span("Today's workout: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_workout}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Distance:", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_km} km", style={'display': 'inline-block', 'width': '150px'}),
        ], style={'display': 'flex', 'align-items': 'baseline'}),
        html.P(children=[
            html.Span("Duration: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_time}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Avg Pace: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{formatted_todays_pace}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
        html.P(children=[
            html.Span("Avg Power: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_watt} W", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Avg HR: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_HFQ}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
        html.P(children=[
            html.Span("Elevation Gain: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_HM} m", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Zone (W)", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_zone_start} - {todays_zone_end}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),        
    ],
    style={'display':'inline-block','width':'45%','border': '1px solid black', 'padding': '10px', 'margin': '10px', 'background-color': 'rgba(178, 34, 34, 0.4)'}  # Adjust width as needed
),

    html.Div(
    children=[    
        html.P(children=[
            html.Span("Kilometers YTD: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{YTD_km_data} km", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Elevation gain YTD: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{YTD_HM_data} m", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(f"{current_year} goal: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{running_goal} km", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("KM with Kids: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_sum_kids} km", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(f"Δ to {current_year} goal: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_goal_difference} km", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(f"KM with Kids {current_year}: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_ytd_km_sum_kids_j} km", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
        html.P(children=[
            html.Span("Today's CP:", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_CP_value} W", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Max CP YTD:", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{max_CP} W", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
    ],
    style={'display':'inline-block','width': '45%', 'border': '1px solid black', 'padding': '10px', 'margin': '10px', 'background-color': 'rgba(64, 162, 111, 0.5)'}  # Adjust width as needed
),

    html.Div(
        children=[    
            html.P(children=[
                html.Span("Tomorrow's workout: ", style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_workout}", style={'display': 'inline-block', 'width': '150px'}),
                html.Span("Tomorrow's Distance:", style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_km} km", style={'display': 'inline-block', 'width': '150px'}),
            ], style={'display': 'flex', 'align-items': 'baseline'}),
            html.P(children=[
                html.Span("Tomorrow's duration: ", style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_time}", style={'display': 'inline-block', 'width': '150px'}),
                html.Span("Zone (W)", style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_zone_start} - {tomorrows_zone_end}", style={'display': 'inline-block', 'width': '150px'})
          ], style={'display': 'flex', 'align-items': 'baseline'}),  
        ],
        style={'display':'inline-block','width': '45%','border': '1px solid black', 'padding': '10px', 'margin': '10px', 'background-color': 'rgba(153, 97, 0, 0.5)'}
        ),

   html.Div(
    children=[    
        html.P(children=[
            html.Span("Today's CTL: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_rCTL_value}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Pwr/Hr (ø 42 d): ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_pwr_hr_42}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
        html.P(children=[
            html.Span("Today's ATL: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_ATL}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Today's load: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_load}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
    ],
    style={'display':'inline-block','width': '45%', 'border': '1px solid black', 'padding': '10px', 'margin': '10px', 'background-color': 'rgba(113, 135, 38, 0.5)'}  # Adjust width as needed
),

html.Div(
    children=[    
        html.P(children=[
            html.Span("5k: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{time_5k} ({CP_5k} W)", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("10k: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{time_10k} ({CP_10k} W)", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span("Half Marathon: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{time_HM} ({CP_HM} W)", style={'display': 'inline-block', 'width': '150px'}),
            html.Span("Marathon: ", style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{time_marathon} ({CP_M} W)", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
    ],
    style={'width': '45%', 'border': '1px solid black', 'padding': '10px', 'margin': '10px', 'background-color': 'rgba(34, 180, 180, 0.5)'}  # Adjust width as needed
),

html.Div(
    style={'width': '45%', 'margin': '10px', 'border': '1px solid black', 'margin-top':'20px', 'padding': '10px'},  # Adjust width as needed
    children=[    
    dash_table.DataTable(
        id='running-km-table',
        columns=[
        {'name': 'Schuh', 'id': 'Schuh'},
        {'name': 'Total Running KM', 'id': 'KM'},
    ],
    data=df_Schuhe,
    style_table={
        'width': '85%',
        'margin': '20px',
        'border': 'none',
    },
    style_cell={
        'textAlign': 'left',
        'whiteSpace': 'normal',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'border': 'none',
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
    ],
    
),
        ]),

# Tab 2 - Power Values
dcc.Tab(label='Power-based Figures', children=[
        html.Div([
            # CTL copmarison
            html.Div(dcc.Graph(figure=fig1, config={'displayModeBar': False})),
            # load comparison
            html.Div(dcc.Graph(figure=fig2, config={'displayModeBar': False})),
        ], style={'display': 'flex'}),
        html.Div([
            # CP vs FTP
            html.Div(dcc.Graph(figure=fig3, config={'displayModeBar': False})),
            # TSS vs RSS per run
            html.Div(dcc.Graph(figure=fig4, config={'displayModeBar': False})),
        ], style={'display': 'flex'}),     
        html.Div([
            # pwr/hr
            html.Div(dcc.Graph(figure=fig9, config={'displayModeBar': False})),
            # rss/tss
            html.Div(dcc.Graph(figure=fig19, config={'displayModeBar': False})),
        ], style={'display': 'flex'}),
        html.Div([
            # CTL vs 42d avg
            html.Div(dcc.Graph(figure=fig20, config={'displayModeBar': False})),
            # rss/tss
            html.Div(dcc.Graph(figure=fig21, config={'displayModeBar': False})),
        ], style={'display': 'flex'}),
        html.Div([
            # CTL und RSS
            html.Div(dcc.Graph(figure=fig1neu, config={'displayModeBar': False})),
            # rss/tss
            html.Div(dcc.Graph(figure=fig2neu, config={'displayModeBar': False})),
        ], style={'display': 'flex'}),
        ]),

# Tab 3 - Runnung Volume
dcc.Tab(label='Running Volume', children=[
        html.Div([
        # weekly km vs RSS
        html.Div(dcc.Graph(figure=fig11, config={'displayModeBar': False})),
        # Goal Graph YTD
        html.Div(dcc.Graph(figure=fig13, config={'displayModeBar': False})),
    ], style={'display': 'flex'}),     
        html.Div([
        # Trailing 90 and 365 day volume
        html.Div(dcc.Graph(figure=fig8, config={'displayModeBar': False})),
        # Yearly Progress
        html.Div(dcc.Graph(figure=fig5, config={'displayModeBar': False})),
    ], style={'display': 'flex'}),
        html.Div([
        # Elevation Gain month
        html.Div(dcc.Graph(figure=fig6, config={'displayModeBar': False})),
        # Elevation Gain year
        html.Div(dcc.Graph(figure=fig16, config={'displayModeBar': False})),
    ], style={'display': 'flex'}), 
        ]),

# Tab 4 - Daten 2024
dcc.Tab(label=f'{current_year}', children=[
        html.Div([
        # Goal Graph for 2024
        html.Div(dcc.Graph(figure=fig15, config={'displayModeBar': False})),
    ], style={'display': 'flex'}),     
        html.Div([
        # Monthly Km for current year
        html.Div(dcc.Graph(figure=fig18, config={'displayModeBar': False})),
        # weekly Km and TSS for current year
        html.Div(dcc.Graph(figure=fig17, config={'displayModeBar': False})),
    ], style={'display': 'flex'}),  
        html.Div([
        # load for current year
            dcc.Graph(figure=fig14, config={'displayModeBar': False}),
    ],),  
        ]),

# Tab 5 - Peaks Map
        dcc.Tab(label='Peaks Map', children=[
            html.Div([
                html.Iframe(
                    srcDoc=open('peaks_progress.html', 'r').read(),
                    width='100%',
                    height='700px',
                ),
            ]),
        ]),
    ]),
])    

if __name__ == '__main__':
    app.run(debug=True)

#if __name__ == "__main__":
#    app.run_server(debug=True, host='0.0.0.0', port=8050)