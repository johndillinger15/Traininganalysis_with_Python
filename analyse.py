from dash import Dash, html, dcc, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Color-scheme
# rgb(178, 34, 34), rgb(153, 97, 0), rgb(113, 135, 38), rgb(64, 162, 111), rgb(34, 180, 180)
# firebrick: rgb(178, 34, 34), complementary blue: rgb(34, 180, 180)
# https://colordesigner.io/gradient-generator/?mode=lch#B22222-22B4B4

# Load dash app
app = Dash(__name__)

# Load data from Excel file
df = pd.read_excel("../training.xlsx", sheet_name="2018+", usecols="C:X")
df_active_shoes = pd.read_excel('../training.xlsx' , sheet_name="Schuhe", usecols="A")
df_active_shoes.dropna(inplace=True)
df_peaks = pd.read_csv("../peaks_projekt/Peaks_Map/peaks_data.csv")
df_raw_peaks = pd.read_csv("../peaks_projekt/Peaks_Map/peaks_raw_data.csv")
df_peaks = df_peaks.filter(['name','elevation','gelaufen'])
df_peaks.index = df_peaks.index +1
# import rowing data
df_row = pd.read_excel("../training.xlsx", sheet_name="rowing")
# Work on a copy to avoid view-vs-copy pitfalls
df_row = df_row.copy()
# Ensure date
df_row['Datum'] = pd.to_datetime(df_row['Datum'], dayfirst=True, errors='coerce')

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
# Year to Date
df_ytd = df[(df['Year'] == current_year) & (df['Date'] <= pd.to_datetime('today'))]
df_ytd = df_ytd.copy()
df_ytd.fillna({'KM':0, 'HFQ':0, 'W1':0, 'W2':0, 'HM':0, 'Pace':0}, inplace=True)
# Year to Tomorrow
df_ytt = df[(df['Date'] >= pd.Timestamp(datetime(tomorrow.year, 1, 1))) & (df['Date'] <= tomorrow)]
df_ytt = df_ytt.copy()  # Explicitly copy the subset of the DataFrame
df_ytt.fillna({'KM':0, 'W1':0, 'W2':0}, inplace=True)
# Only including data since 2020
df_since_2020 = df[df['Year'] >= 2020]
# Limiting data to today
df_until_today = df[df['Date'] <= today]
# Only including data for current year
df_current_year = df[df['Date'].dt.year == current_year]
df_current_year_filtered = df_current_year[['Date', 'KM']].copy()
df_current_year_filtered.fillna({'KM':0}, inplace=True)
df_current_year_filtered.loc[:, 'actual'] = df_current_year_filtered['KM'].cumsum()
# Calculate the date 90 days ago
ninety_days_ago = today2 - timedelta(days=90)
one_year_ago = today2 - timedelta(days=365)
# Filter the dataframe for the last 90 days and excluding future dates
df_last_90_days = df[(df['Date'] >= ninety_days_ago) & (df['Date'] <= today2)]
df_last_365_days = df[(df['Date'] >= one_year_ago) & (df['Date'] <= today2)]

# Calculate the date 7 days ago
seven_days_ago = today2 - timedelta(days=7)
# Filter the dataframe for the last 7 days and excluding future dates
df_last_7_days = df[(df['Date'] >= seven_days_ago) & (df['Date'] <= today2)].copy()
# Convert 'Zeit' column to timedelta
df_last_7_days.loc[:, 'Zeit'] = pd.to_timedelta(df_last_7_days['Zeit'].astype(str))
# Drop rows with missing 'Zeit'
df_last_7_days.dropna(subset=['Zeit'], inplace=True)

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
df_Schuhe = df_Schuhe[df_Schuhe['Schuh'].isin(df_active_shoes['Schuh'])]
df_Schuhe['KM'] = df_Schuhe['KM'].round(0)
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
cumulative_sum_90_today = merged_df['Cumulative_Sum_90_Days'].iloc[-1]
cumulative_sum_365_today = merged_df['Cumulative_Sum_365_Days'].iloc[-1]
cumulative_sum_90_today = int(round(cumulative_sum_90_today, 0))
cumulative_sum_365_today = int(round(cumulative_sum_365_today, 0))

# PWR/HR
# Calculate the cumulative average of 'pwr/hr' for each day starting from January 1, 2024, for the trailing 42 days
cumulative_avg_pwr_hr_42_days = []
for date in pd.date_range(start_date, pd.to_datetime('today')):
    trailing_42_days_avg_pwr_hr = df[(df['Date'] >= (date - pd.DateOffset(days=41))) & (df['Date'] <= date)]['pwr/hr'].mean()
    cumulative_avg_pwr_hr_42_days.append(trailing_42_days_avg_pwr_hr)
# Create a DataFrame for the results
result_df_avg_pwr_hr_42_days = pd.DataFrame({'Date': pd.date_range(start_date, pd.to_datetime('today')), 'Cumulative_Avg_pwr_hr_42_Days': cumulative_avg_pwr_hr_42_days})
# Get today's and yesterday's values
todays_value = result_df_avg_pwr_hr_42_days['Cumulative_Avg_pwr_hr_42_Days'].iloc[-1]
yesterdays_value = result_df_avg_pwr_hr_42_days['Cumulative_Avg_pwr_hr_42_Days'].iloc[-2]
# Compare today's value with yesterday's and store the arrow in a variable
pwr_hr_arrow = '▲' if todays_value > yesterdays_value else '▼' if todays_value < yesterdays_value else '='

# Changing decimals of numbers in dataframes
# Rounding the last value in the 'CP' column to 0 decimal places
rounded_CP_value = int(round(df_ytd['CP'].iloc[-1], 0))

# Rounding the last value in the 'RSS load' column to 2 decimal places
rounded_load_value= round(df_ytd['load'].iloc[-1], 2)

# Rounding the last value in the 'rCTL' column to 1 decimal place
rounded_rCTL_value= round(df_ytd['CTL'].iloc[-1], 1)

# Rounding the trailing 90d km value to 0 decimal place
YTD_km_data = int(round(df_ytd['KM'].sum()))

# Rounding trailing 42d avg of Pwr/HR to 2 decimal places
rounded_pwr_hr_42 = round(result_df_avg_pwr_hr_42_days['Cumulative_Avg_pwr_hr_42_Days'].iloc[-1],2)

# Define variables for Dashboard
todays_workout = df_ytd['Art'].iloc[-1]
todays_km = df_ytd['KM'].iloc[-1]
todays_km = round(todays_km, 2)
todays_time = df_ytd['Zeit'].iloc[-1]
todays_pace = df_ytd['Pace'].iloc[-1]
todays_watt = df_ytd['Watt'].iloc[-1]
todays_HFQ = df_ytd['HFQ'].iloc[-1]
todays_HFQ = int(round(todays_HFQ, 0))
todays_HM = df_ytd['HM'].iloc[-1]
todays_HM = int(round(todays_HM, 0))
todays_ATL = df_ytd['ATL'].iloc[-1]
todays_ATL = round(todays_ATL, 1)
todays_load = df_ytd['load'].iloc[-1]
todays_pwr_hr = df_ytd['pwr/hr'].iloc[-1]
todays_pwr_hr = round(todays_pwr_hr, 2)
yesterdays_load = df_ytd['load'].iloc[-2]
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
max_pwr_hr = result_df_avg_pwr_hr_42_days['Cumulative_Avg_pwr_hr_42_Days'].max()
max_pwr_hr = round(max_pwr_hr, 2)
this_weeks_KM = weekly_data['KM'].iloc[-1]
this_weeks_RSS = weekly_data['RSS'].iloc[-1]
last_weeks_KM = weekly_data['KM'].iloc[-2]
last_weeks_RSS = weekly_data['RSS'].iloc[-2]
last_7d_KM = df_last_7_days['KM'].sum()
last_7d_KM_rounded = int(round(last_7d_KM, 0))
last_7d_RSS = df_last_7_days['RSS'].sum()
last_7d_RSS = int(round(last_7d_RSS, 0))
last_7d_time = df_last_7_days['Zeit'].sum()
weekly_average_KM = weekly_data['KM'].mean()
weekly_average_RSS = weekly_data['RSS'].mean()
peaks_completed = len(df_peaks)
peaks_total = len(df_raw_peaks)
peaks_percentage = (peaks_completed / peaks_total)*100
peaks_percentage = round(peaks_percentage, 2)
longrun_7d_km = df_last_7_days['KM'].max()

# 1) Clean inputs and aggregate to one row per day
row_km = df_row[['Datum', 'KM']].copy()
row_km['Datum'] = pd.to_datetime(row_km['Datum'], dayfirst=True, errors='coerce').dt.floor('D')
row_km['KM'] = (
    row_km['KM']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.extract(r'([-+]?\d*\.?\d+)', expand=False)
        .astype(float)
        .fillna(0.0)
)
row_km = row_km.dropna(subset=['Datum'])
row_km = row_km[row_km['Datum'] <= pd.Timestamp.today().normalize()]
# 2) Aggregate daily totals (in case of multiple rowing sessions per day)
row_km = row_km.groupby('Datum', as_index=False)['KM'].sum()
# 3) Rolling windows
today     = pd.Timestamp.today().normalize()
start_90  = today - pd.Timedelta(days=90)
start_365 = today - pd.Timedelta(days=365)
row_vol_90  = int(round(row_km.loc[row_km['Datum'] >= start_90,  'KM'].sum(), 0))
row_vol_365 = int(round(row_km.loc[row_km['Datum'] >= start_365, 'KM'].sum(), 0))



# Check if last_7d_KM is zero
if last_7d_KM != 0:
    percent_longrun = (longrun_7d_km / last_7d_KM) * 100
    percent_longrun = int(round(percent_longrun, 0))
else:
    # Handle the case where last_7d_KM is zero
    percent_longrun = 0  # Set to a default value or handle it accordingly

if last_7d_time != 0:
    # fix last 7day time
    hours_7d = last_7d_time.seconds // 3600
    minutes_7d = (last_7d_time.seconds % 3600) // 60
    seconds_7d = last_7d_time.seconds % 60
    last_7d_time = f"{hours_7d:02d}:{minutes_7d:02d}:{seconds_7d:02d}"
else:
    last_7d_time = 0

# Define arrows
load_arrow = '▲' if todays_load > yesterdays_load else '▼' if todays_load < yesterdays_load else '='
weekly_arrow_KM = '▲' if last_weeks_KM > weekly_average_KM else '▼' if last_weeks_KM < weekly_average_KM else '='
arrow_7d_KM = '▲' if last_7d_KM > weekly_average_KM else '▼' if last_7d_KM < weekly_average_KM else '='
weekly_arrow_RSS = '▲' if last_weeks_RSS > weekly_average_RSS else '▼' if last_weeks_RSS < weekly_average_RSS else '='
arrow_7d_RSS = '▲' if last_7d_RSS > weekly_average_RSS else '▼' if last_7d_RSS < weekly_average_RSS else '='


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
rounded_ytd_km_sum_kids_j = int(round(ytd_km_sum_kids_j['KM'].sum(), 0))

# Calculate running times
# Distances in meters
distance_5k = 5000
distance_10k = 10000
distance_HM = 21097.5  # Half Marathon
distance_marathon = 42195  # Marathon

# % of CP for each distance
modification_factor_5k = 1.065
modification_factor_10k = 1.017
modification_factor_HM = 0.96
modification_factor_marathon = 0.9

# calculate target race CP
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

# ECOR for each distance
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
# RSS per run and CTL
fig1 = px.bar(df_last_90_days, x='Date', y=['RSS'], barmode='group', text='Art',
              labels={'value':'RSS'}, color_discrete_sequence=['#2283B4'])

fig1.add_trace(go.Scatter(x=df_last_90_days['Date'], y=df_last_90_days['CTL'], mode='lines', line_width=3, name='CTL', line=dict(color='firebrick')))

fig1.update_layout(
    title='RSS and CTL for last 90d (fig1)',
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)
  
# Load for last 90d
fig2 = px.line(df_last_90_days, x='Date', y=['load'],
              labels={'value': 'load', 'variable': 'load'}, 
              line_shape='linear', color_discrete_sequence=['#2283B4','firebrick'], range_y=[0,2])

fig2.update_layout(
    title='load for last 90d (fig2)',
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=False)

# Define the shading ranges
shading_ranges_new = [
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 0.5, 'y1': 0.8, 'color': 'rgba(53, 77, 115,0.5)'},  # blue
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 0.8, 'y1': 1, 'color': 'rgba(0, 200, 0,0.5)'},  # lightgreen
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 1, 'y1': 1.3, 'color': 'rgba(0,255,100,0.5)'},  # darkgreen
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 1.3, 'y1': 1.5, 'color' : 'rgba(255,255,0,0.5)'},  # Yellow
    {'x0': df_last_90_days['Date'].min(), 'x1': df_last_90_days['Date'].max(), 'y0': 1.5, 'y1': 2, 'color': 'rgba(255,0,0,0.5)'}  # Red
]
for shading_range in shading_ranges_new:
    fig2.add_shape(type="rect",
                  x0=shading_range['x0'], x1=shading_range['x1'],
                  y0=shading_range['y0'], y1=shading_range['y1'],
                  fillcolor=shading_range['color'], opacity=0.5, layer='below', line_width=0)


# Create the primary line chart for 'CP'
fig3 = px.line(df_last_365_days, x='Date', y=['CP'], 
               labels={'value': 'CP/FTP', 'variable': 'Metric'},
               line_shape='linear', color_discrete_sequence=['#2283B4'])

# Update layout of the figure for primary axis
fig3.update_layout(
    title='CP for last 365 days (fig3)',
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)


# Monthly running progress

# Extract the year and group by year, summing up the kilometers
df_running_progress = df_until_today.groupby(df_until_today['Date'].dt.year).agg({'KM': 'sum'}).reset_index()
df_running_progress.rename(columns={'Date': 'Year'}, inplace=True)

# Create a new column to categorize the 'KM' values
def categorize_km(KM):
    if KM < 1000:
        return '0-999 km'
    elif 1000 <= KM < 2000:
        return '1000-1999 km'
    else:
        return '>2000 km'

df_running_progress['Category'] = df_running_progress['KM'].apply(categorize_km)

# Define color mapping for categories
color_map = {
    '0-999 km': 'rgb(34, 180, 180)',    # Light teal
    '1000-1999 km': 'rgb(113, 135, 38)', # Orange
    '>2000 km': 'firebrick'      # Crimson
}

# Create the histogram
fig5 = px.bar(
    df_running_progress, 
    x='Year', 
    y='KM', 
    color='Category',
    color_discrete_map=color_map
)

# Update layout
fig5.update_layout(
    title=dict(
        text='Yearly Running Volume (fig5)',
        xanchor='left', 
        yanchor='top'
    ),
    width=650,
    height=400,
    showlegend=False,
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    bargap=0.2,
    plot_bgcolor="white",
    yaxis_title='KM',
)

# Add threshold lines
fig5.add_hline(y=1000, line_width=1, line_dash='dash', line_color="firebrick")
fig5.add_hline(y=2000, line_width=1, line_dash='dash', line_color="firebrick")


# Monthly elevation gain

# Define the order of months
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the histogram
fig6 = px.histogram(df_since_2020, x='Month', y='HM', color='Year', barmode='group', category_orders={'Month': month_order}, color_discrete_sequence=['rgb(178, 34, 34)', 'rgb(153, 97, 0)', 'rgb(113, 135, 38)', 'rgb(64, 162, 111)', 'rgb(34, 180, 180)'])

fig6.update_layout(
    title='Elevation Gain Per Month (fig6)',
    width=650,  # Set the width of the graph
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
fig8 = make_subplots(specs=[[{"secondary_y": True}]])

fig8.add_trace(
    go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_90_Days'],
               mode='lines', name='Trailing 90days km', line=dict(color='firebrick')),
    secondary_y=False
)

fig8.add_trace(
    go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_365_Days'],
               mode='lines', name='Trailing 365days km', line=dict(color='rgb(34, 180, 180)')),
    secondary_y=True
)

# Layout (note: no legacy titlefont anywhere)
fig8.update_layout(
    title=dict(text='Trailing 90d and 365d Running Volume (fig8)'),
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title_text=None,
    width=650,
    height=400,
    plot_bgcolor="white"
)

# Axis titles and fonts (new API)
fig8.update_yaxes(
    title_text='Trailing 90 Days',
    title_font=dict(color="firebrick"),
    tickfont=dict(color="firebrick"),
    secondary_y=False
)
fig8.update_yaxes(
    title_text='Trailing 365 Days',
    title_font=dict(color="rgb(34, 180, 180)"),
    tickfont=dict(color="rgb(34, 180, 180)"),
    secondary_y=True
)

# add shaded region
# fig8.add_vrect(x0="2023-10-28", x1="2023-12-28", fillcolor="firebrick", line_width=0, opacity=0.2)

# PWR/HR Graph
fig9 = px.line(result_df_avg_pwr_hr_42_days, x='Date', y='Cumulative_Avg_pwr_hr_42_Days',
              labels={'Cumulative_Avg_pwr_hr_90_Days': 'Avg pwr/hr'}, color_discrete_sequence=['#2283B4'])

fig9.update_layout(
    title='Avg of pwr/hr for Trailing 42 Days (fig9)',
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    yaxis_title='Pwr/Hr',
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# Weekly Data last 365d
fig11 = px.bar(
    weekly_data, x='Date', y='KM',
    labels={'KM': 'KM'},
    color_discrete_sequence=['rgb(34, 180, 180)']
)

# Add RSS as scatter on secondary axis
fig11.add_trace(
    px.scatter(
        weekly_data, x='Date', y='RSS',
        labels={'RSS': 'RSS'},
        color_discrete_sequence=['firebrick']
    ).update_traces(yaxis='y2').data[0]
)

ymax1 = weekly_data['RSS'].max() * 1.10

fig11.update_layout(
    title=dict(text='Weekly KM vs RSS (fig11)'),
    # Primary y-axis (KM)
    yaxis=dict(
        range=[0, ymax1/5],
        title=dict(text='KM', font=dict(color="#2283B4")),
        tickfont=dict(color="#2283B4")
    ),
    # Secondary y-axis (RSS)
    yaxis2=dict(
        title=dict(text='RSS', font=dict(color="firebrick")),
        tickfont=dict(color="firebrick"),
        overlaying='y',
        side='right',
        range=[0, ymax1]
    ),
    width=650,
    height=400,
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title_text=None,
    plot_bgcolor="white"
)

# Threshold lines on the RSS axis (use yref='y2')
fig11.add_hline(y=30, line=dict(color="black", width=1, dash="dash"), yref="y1")
fig11.add_hline(y=40, line=dict(color="black", width=1, dash="dash"), yref="y1")


# Yearly YTD running goal graph
fig13 = px.line(merged_df_goal, x='Date', y=['actual', 'goal'],  labels={'value':'KM'}, color_discrete_sequence=['rgb(34, 180, 180)','firebrick'])

fig13.update_layout(
    title=f'YTD Progress in {current_year} vs Goal (KM) (fig13)',
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white",
)

# 2024 monthly kilometer
fig14 = px.histogram(df_current_year, x='Month', y='KM', barmode='group', color_discrete_sequence=['rgb(34, 180, 180)'])

fig14.update_layout(
    title=f'Monthly Volumes for {current_year} (fig14)',
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1, # gap between bars of the same location coordinates
    plot_bgcolor="white",    
    yaxis_title='KM',
)
fig14.update_xaxes(
    tickvals=list(range(1, 13)),
    ticktext=['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December'],
    ticklabelmode='period'
)

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
    title=dict(text=f'Goal vs actual {current_year} (fig15)'),
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
    plot_bgcolor="white",
)

fig15.add_vline(x=f"{today}", line_width=2, line_color="rgb(153, 97, 0)")


# elevation gain per year
fig16 = px.histogram(df_since_2020, x='Year', y='HM', color='Year', color_discrete_sequence=['rgb(178, 34, 34)', 'rgb(153, 97, 0)', 'rgb(113, 135, 38)', 'rgb(64, 162, 111)', 'rgb(34, 180, 180)'])

fig16.update_layout(
    title='Elevation Gain Per Year (fig16)',
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1, # gap between bars of the same location coordinates
    plot_bgcolor="white",    
    yaxis_title='HM',
)

fig16.add_hline(y=10000, line_width=1, line_dash='dash', line_color="firebrick")

# Weekly Data current year
fig17 = px.bar(
    weekly_data_current_year,
    x='Date',
    y='KM',
    labels={'KM': 'KM'},
    color_discrete_sequence=['rgb(34, 180, 180)']
)

# Add RSS on secondary y-axis
fig17.add_trace(
    px.scatter(
        weekly_data_current_year,
        x='Date',
        y='RSS',
        labels={'RSS': 'RSS'},
        color_discrete_sequence=['firebrick']
    ).update_traces(yaxis='y2').data[0]
)

ymax2 = weekly_data_current_year['RSS'].max() * 1.10

fig17.update_layout(
    title=dict(text=f'Weekly KM vs RSS for {current_year} (fig17)'),
    # Primary y-axis (KM)
    yaxis=dict(
        range=[0, ymax2/5],
        title=dict(text='KM', font=dict(color="#2283B4")),
        tickfont=dict(color="#2283B4")
    ),
    # Secondary y-axis (RSS)
    yaxis2=dict(
        title=dict(text='RSS', font=dict(color="firebrick")),
        tickfont=dict(color="firebrick"),
        overlaying='y',
        side='right',
        range=[0, ymax2]
    ),
    width=650,
    height=400,
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title_text=None,
    plot_bgcolor="white"
)

# Reference lines
fig17.add_vline(x=f"{today}", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")
fig17.add_hline(y=30, line=dict(color="black", width=1, dash="dash"), yref="y1")
fig17.add_hline(y=40, line=dict(color="black", width=1, dash="dash"), yref="y1")


# This year's load
fig18 = px.line(df_current_year, x='Date', y=['load'],
              labels={'value': 'load', 'variable': 'load'},
              line_shape='linear', color_discrete_sequence=['firebrick'])

fig18.update_layout(
    title=dict(text=f'Load comparison of {current_year} (fig18)'),
    yaxis=dict(range=[0,2]),
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    showlegend=False,
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


# ---- Running Calendar Heatmap (365d, Mon on top; hue by Art, intensity by RSS) — fig23 ----
# Requires: import numpy as np

# 0) Prepare data (one run per day)
df_run = df[['Date','RSS','KM','Art']].copy()
df_run['Date'] = pd.to_datetime(df_run['Date'], errors='coerce')
df_run['RSS']  = pd.to_numeric(df_run['RSS'], errors='coerce').fillna(0)
df_run['KM']   = pd.to_numeric(df_run['KM'], errors='coerce').fillna(0)
df_run['Art']  = df_run['Art'].fillna('')

# 1) Window = last 365 days; align grid start to Monday
end = pd.Timestamp.today().normalize()
start = end - pd.Timedelta(days=364)
grid_start = start - pd.Timedelta(days=start.weekday())
df_run = df_run[(df_run['Date'] >= start) & (df_run['Date'] <= end)].copy()

# 2) Bucketing function for coloring (keep categories LOWERCASE)
def _cat(a: str):
    a_low = str(a).lower()
    if 'z2' in a_low:       return 'z2'
    if 'trail' in a_low:    return 'trail'
    if 'wk' in a_low:       return 'wk'
    if 'row' in a_low:      return 'row'
    return 'other'

df_run['cat'] = df_run['Art'].apply(_cat)

# 3) Build full calendar (inactive days explicit)
all_days = pd.DataFrame({'Date': pd.date_range(grid_start, end, freq='D')})
cal = all_days.merge(df_run, on='Date', how='left')
cal[['RSS','KM']] = cal[['RSS','KM']].fillna(0)
cal['cat'] = cal['cat'].fillna('none')
cal['Art'] = cal['Art'].fillna('')  # keep original Art string for hover

# 4) Grid coordinates
cal['week'] = ((cal['Date'] - grid_start).dt.days // 7).astype(int)
cal['dow']  = cal['Date'].dt.weekday  # Mon=0..Sun=6

weeks = list(range(cal['week'].min(), cal['week'].max() + 1))
rows_order = [0,1,2,3,4,5,6]
y_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# 5) Pivot to matrices
date_df = (cal.pivot(index='dow', columns='week', values='Date')
             .reindex(index=rows_order, columns=weeks))
rss_mat = (cal.pivot(index='dow', columns='week', values='RSS')
             .reindex(index=rows_order, columns=weeks, fill_value=0)).values
km_mat  = (cal.pivot(index='dow', columns='week', values='KM')
             .reindex(index=rows_order, columns=weeks, fill_value=0)).values
cat_df  = (cal.pivot(index='dow', columns='week', values='cat')
             .reindex(index=rows_order, columns=weeks).fillna('none'))
art_df  = (cal.pivot(index='dow', columns='week', values='Art')
             .reindex(index=rows_order, columns=weeks).fillna(''))

# Pre-format date strings to avoid hover NaNs
date_str_mat = date_df.map(lambda d: '' if pd.isna(d) else d.strftime('%Y-%m-%d')).values
cat_mat = cat_df.values
art_mat = art_df.values

# 6) Normalize intensity by yearly max RSS
rss_max = max(1.0, float(cal['RSS'].max()))
z_norm_base = rss_mat / rss_max  # 0..1

# 7) Colors and categories (Option 2: intervals = firebrick, race = crimson)
colors = {
    'z2':    (34, 180, 180),   # teal
    'trail': (255, 162, 0),    # orange
    'other': (178, 34, 34),    # firebrick
    'wk':    (190, 37, 186),   # violet
    'row':   (0, 102, 204),    # blue
}
cats = ['z2','trail','other','wk', 'row']



# 8) Build figure: one heatmap trace per category (use NaN outside the category!)
fig23 = go.Figure()

for cat in cats:
    r, g, b = colors[cat]
    mask_bool = (cat_mat == cat)

    if cat == 'row':
        # constant 0.6 opacity blue
        z_norm = np.where(mask_bool, 1.0, np.nan)
        colorscale = [[0.0, f'rgba({r},{g},{b},0.6)'],
                      [1.0, f'rgba({r},{g},{b},0.6)']]
        
    else:
        # intensity scaled by RSS
        z_norm = np.where(mask_bool, z_norm_base, np.nan)
        colorscale = [
            [0.00, 'rgb(255,255,255)'],
            [0.01, f'rgba({r},{g},{b},0.15)'],
            [0.20, f'rgba({r},{g},{b},0.35)'],
            [0.60, f'rgba({r},{g},{b},0.60)'],
            [1.00, f'rgb({r},{g},{b})'],
        ]

    # customdata unchanged
    cd = np.dstack([date_str_mat, art_mat, rss_mat, km_mat])


    fig23.add_trace(go.Heatmap(
        z=z_norm,
        x=weeks,
        y=y_labels,
        zmin=0, zmax=1,
        colorscale=colorscale,
        showscale=False,
        customdata=cd,
        hovertemplate=(
            "Date: %{customdata[0]}<br>"
            "Art: %{customdata[1]}<br>"
            "RSS: %{customdata[2]:.0f}<br>"
            "KM: %{customdata[3]:.2f}<extra></extra>"
        )
    ))


# 9) Month labels
week_starts = [grid_start + pd.Timedelta(days=int(w)*7) for w in weeks]
tickvals = weeks
ticktext, prev_month = [], None
for d in week_starts:
    ticktext.append(d.strftime('%b') if (prev_month is None or d.month != prev_month) else '')
    prev_month = d.month

# 10) Layout
fig23.update_layout(
    height=130,
    width=1200,
    plot_bgcolor="white",
    margin=dict(l=0, r=0, b=0, t=0),
    hovermode='closest'
)
fig23.update_yaxes(autorange='reversed', showticklabels=True, showgrid=False, zeroline=False)
fig23.update_xaxes(
    tickmode='array', tickvals=tickvals, ticktext=ticktext,
    showticklabels=True, showgrid=False, zeroline=False
)

###
# Rowing
###

# Normalize German decimals and cast numerics
for col in ['KM', 'Watt', 'HFQ', 'pwr/hr']:
    if col in df_row.columns:
        df_row[col] = (
            df_row[col]
            .astype(str)
            .str.replace(',', '.', regex=False)  # decimal comma -> dot
        )
        df_row[col] = pd.to_numeric(df_row[col], errors='coerce')

# Convert Zeit to minutes without needing datetime imports
def _to_minutes(x):
    # treat NaN early
    if pd.isna(x):
        return float('nan')
    # already numeric minutes
    if isinstance(x, (int, float)):
        return float(x)
    # objects like datetime.time: detect via attributes
    if hasattr(x, 'hour') and hasattr(x, 'minute'):
        sec = getattr(x, 'second', 0)
        return x.hour*60 + x.minute + sec/60
    # strings like "mm:ss" or "hh:mm:ss"
    s = str(x).strip()
    if ':' in s:
        parts = s.split(':')
        try:
            if len(parts) == 2:
                m, sec = parts
                return int(m) + int(sec)/60
            if len(parts) == 3:
                h, m, sec = parts
                return int(h)*60 + int(m) + int(sec)/60
        except Exception:
            pass
    # last resort: numeric coercion
    return pd.to_numeric(s, errors='coerce')

# Build/ensure Zeit_min
if 'Zeit_min' not in df_row.columns:
    df_row['Zeit_min'] = df_row['Zeit'].apply(_to_minutes)
else:
    df_row['Zeit_min'] = pd.to_numeric(df_row['Zeit_min'], errors='coerce')


###
# Rowing calendar heatmap (365d) with WOD shaded by duration in firebrick
# - One workout per day (no per-day grouping)
# - Steady rows shaded in teal by minutes
# - WOD rows shaded in firebrick by minutes
###

# Load sheet
df_row = pd.read_excel("../training.xlsx", sheet_name="rowing").copy()

# Ensure date
df_row['Datum'] = pd.to_datetime(df_row['Datum'], dayfirst=True, errors='coerce')

# Normalize German decimals and cast numerics
for col in ['KM', 'Watt', 'HFQ', 'pwr/hr']:
    if col in df_row.columns:
        df_row[col] = (
            df_row[col]
            .astype(str)
            .str.replace(',', '.', regex=False)  # decimal comma -> dot
        )
        df_row[col] = pd.to_numeric(df_row[col], errors='coerce')

# Convert Zeit to minutes
def _to_minutes(x):
    if pd.isna(x):
        return float('nan')
    # numeric already
    if isinstance(x, (int, float)):
        return float(x)
    # time-like object
    if hasattr(x, 'hour') and hasattr(x, 'minute'):
        sec = getattr(x, 'second', 0)
        return x.hour*60 + x.minute + sec/60
    # strings like "mm:ss" or "hh:mm:ss"
    s = str(x).strip()
    if ':' in s:
        parts = s.split(':')
        try:
            if len(parts) == 2:   # mm:ss
                m, sec = parts
                return int(m) + int(sec)/60
            if len(parts) == 3:   # hh:mm:ss
                h, m, sec = parts
                return int(h)*60 + int(m) + int(sec)/60
        except Exception:
            return float('nan')
    # last resort: numeric coercion
    return pd.to_numeric(s, errors='coerce')

# Ensure Zeit_min
if 'Zeit_min' not in df_row.columns:
    df_row['Zeit_min'] = df_row['Zeit'].apply(_to_minutes)
else:
    df_row['Zeit_min'] = pd.to_numeric(df_row['Zeit_min'], errors='coerce')

# Pretty time string
def _fmt_minutes(m):
    if pd.isna(m): return ''
    m = float(m)
    h = int(m // 60)
    mm = int(round(m - 60*h))
    return f"{h}:{mm:02d}h" if h > 0 else f"{mm} min"

# Build working frame
row = df_row[['Datum','Art','KM'] + ([c for c in ['Zeit_min','Zeit'] if c in df_row.columns])].copy()
row['KM']  = pd.to_numeric(row['KM'], errors='coerce').fillna(0)
row['Art'] = row['Art'].fillna('')

# Ensure Zeit_min and Zeit_str
if 'Zeit_min' not in row.columns:
    row['Zeit_min'] = row['Zeit'].apply(_to_minutes)
else:
    row['Zeit_min'] = pd.to_numeric(row['Zeit_min'], errors='coerce')

if 'Zeit' in row.columns and row['Zeit'].notna().any():
    row['Zeit_str'] = row['Zeit'].fillna('').astype(str)
else:
    row['Zeit_str'] = row['Zeit_min'].apply(_fmt_minutes)

# 365d window aligned to Monday
end = pd.Timestamp.today().normalize()
start = end - pd.Timedelta(days=364)
grid_start = start - pd.Timedelta(days=start.weekday())

# Keep only window (one workout per day by design)
row = row[(row['Datum'] >= start) & (row['Datum'] <= end)].copy()

# WOD flag
row['is_wod'] = row['Art'].astype(str).str.strip().str.upper().eq('WOD')

# Build full calendar grid (include inactive days)
all_days = pd.DataFrame({'Datum': pd.date_range(grid_start, end, freq='D')})
cal = all_days.merge(
    row[['Datum','Art','KM','Zeit_min','Zeit_str','is_wod']],
    on='Datum', how='left'
)

# Make dtypes explicit (avoids future downcasting warnings)
cal['KM']       = pd.to_numeric(cal['KM'], errors='coerce')
cal['Zeit_min'] = pd.to_numeric(cal['Zeit_min'], errors='coerce')
cal['Art']      = cal['Art'].astype('string')
cal['Zeit_str'] = cal['Zeit_str'].astype('string')
cal['is_wod']   = cal['is_wod'].astype('boolean')  # pandas nullable boolean

# Now fill NA with intended defaults
cal['KM']       = cal['KM'].fillna(0)
cal['Zeit_min'] = cal['Zeit_min'].fillna(0)
cal['Art']      = cal['Art'].fillna('')
cal['Zeit_str'] = cal['Zeit_str'].fillna('')
cal['is_wod']   = cal['is_wod'].fillna(False)


# Grid coordinates
cal['week'] = ((cal['Datum'] - grid_start).dt.days // 7).astype(int)  # columns
cal['dow']  = cal['Datum'].dt.weekday                                 # rows (Mon..Sun)

weeks = list(range(cal['week'].min(), cal['week'].max() + 1))
rows_order = [0,1,2,3,4,5,6]
y_labels   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# Pivot to matrices
date_df = (cal.pivot(index='dow', columns='week', values='Datum')
             .reindex(index=rows_order, columns=weeks))
km_mat   = (cal.pivot(index='dow', columns='week', values='KM')
              .reindex(index=rows_order, columns=weeks, fill_value=0)).values
min_mat  = (cal.pivot(index='dow', columns='week', values='Zeit_min')
              .reindex(index=rows_order, columns=weeks, fill_value=0)).values
art_mat  = (cal.pivot(index='dow', columns='week', values='Art')
              .reindex(index=rows_order, columns=weeks).fillna('')).values
zeit_txt = (cal.pivot(index='dow', columns='week', values='Zeit_str')
              .reindex(index=rows_order, columns=weeks).fillna('')).values
wod_mat  = (cal.pivot(index='dow', columns='week', values='is_wod')
              .reindex(index=rows_order, columns=weeks).fillna(False)).values

date_str = date_df.map(lambda d: '' if pd.isna(d) else d.strftime('%Y-%m-%d')).values

# Intensity scaling (minutes)
mmax = max(1.0, float(cal['Zeit_min'].max()))
z_norm = min_mat / mmax  # 0..1

# Split into two layers: steady (teal) vs WOD (firebrick)
mask_wod    = wod_mat.astype(bool)
mask_steady = ~mask_wod

z_teal = np.where(mask_steady, z_norm, np.nan)
z_red  = np.where(mask_wod,     z_norm, np.nan)

# Colorscales
teal = (34, 180, 180)
colorscale_teal = [
    [0.00, 'rgb(255,255,255)'],
    [0.05, f'rgba({teal[0]},{teal[1]},{teal[2]},0.15)'],
    [0.25, f'rgba({teal[0]},{teal[1]},{teal[2]},0.35)'],
    [0.60, f'rgba({teal[0]},{teal[1]},{teal[2]},0.60)'],
    [1.00, f'rgb({teal[0]},{teal[1]},{teal[2]})'],
]
colorscale_firebrick = [
    [0.00, 'rgb(255,255,255)'],
    [0.05, 'rgba(178,34,34,0.15)'],
    [0.25, 'rgba(178,34,34,0.35)'],
    [0.60, 'rgba(178,34,34,0.60)'],
    [1.00, 'firebrick'],
]

# Hover payload
custom = np.dstack([date_str, art_mat, km_mat, zeit_txt])

# Figure with two heatmap layers
fig_row1 = go.Figure()

# Steady rows (teal)
fig_row1.add_trace(go.Heatmap(
    z=z_teal,
    x=weeks, y=y_labels,
    zmin=0, zmax=1,
    colorscale=colorscale_teal,
    showscale=False,
    name="Steady",
    customdata=custom,
    hovertemplate=("Datum: %{customdata[0]}<br>"
                   "Art: %{customdata[1]}<br>"
                   "Distance: %{customdata[2]:.2f} km<br>"
                   "Zeit: %{customdata[3]}<extra></extra>")
))

# WOD rows (firebrick)
fig_row1.add_trace(go.Heatmap(
    z=z_red,
    x=weeks, y=y_labels,
    zmin=0, zmax=1,
    colorscale=colorscale_firebrick,
    showscale=False,
    name="WOD",
    customdata=custom,
    hovertemplate=("Datum: %{customdata[0]}<br>"
                   "Art: %{customdata[1]}<br>"
                   "Distance: %{customdata[2]:.2f} km<br>"
                   "Zeit: %{customdata[3]}<extra></extra>")
))

# Month labels (first week of each month)
week_starts = [grid_start + pd.Timedelta(days=int(w)*7) for w in weeks]
tickvals, ticktext, prev_month = weeks, [], None
for d in week_starts:
    ticktext.append(d.strftime('%b') if (prev_month is None or d.month != prev_month) else '')
    prev_month = d.month

# Layout
fig_row1.update_layout(
    height=130,
    width=1200,
    plot_bgcolor="white",
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
)
fig_row1.update_yaxes(autorange='reversed', showticklabels=True, showgrid=False, zeroline=False)
fig_row1.update_xaxes(
    tickmode='array', tickvals=tickvals, ticktext=ticktext,
    showticklabels=True, showgrid=False, zeroline=False
)

# ---- Rowing Rolling Volume (90d & 365d) — robust full-history rolls (fig_row2) ----

# 1) Clean inputs and aggregate to one row per day
row_km = df_row[['Datum', 'KM']].copy()
row_km['Datum'] = pd.to_datetime(row_km['Datum'], dayfirst=True, errors='coerce').dt.floor('D')
row_km['KM']    = pd.to_numeric(row_km['KM'], errors='coerce')
row_km = row_km.dropna(subset=['Datum'])
row_km = row_km[row_km['Datum'] <= today]  # ignore future rows

daily_row_agg = (row_km
                 .groupby('Datum', as_index=False)['KM']
                 .sum())

if not daily_row_agg.empty:
    # 2) Build a continuous daily index from first rowing day to today
    full_idx_row = pd.date_range(daily_row_agg['Datum'].min().normalize(), today, freq='D')
    daily_row = (daily_row_agg.set_index('Datum')
                              .reindex(full_idx_row, fill_value=0.0)
                              .rename_axis('Datum')
                              .rename(columns={'KM': 'KM_day'}))

    # 3) Rolling windows via cumulative sums (no dips)
    daily_row['KM_cum'] = daily_row['KM_day'].cumsum()
    daily_row['KM_roll_90']  = (daily_row['KM_cum'] - daily_row['KM_cum'].shift(90,  fill_value=0.0)).clip(lower=0)
    daily_row['KM_roll_365'] = (daily_row['KM_cum'] - daily_row['KM_cum'].shift(365, fill_value=0.0)).clip(lower=0)

    # 4) Slice display to last 365 days (rolls still from full history)
    daily_row_365 = daily_row[daily_row.index >= (today2 - pd.Timedelta(days=365))].reset_index()

# 5) Figure — fig_row2 with secondary_y=True, fig8-style
fig_row2 = make_subplots(specs=[[{"secondary_y": True}]])

# 90d rolling (left y-axis)
fig_row2.add_trace(
    go.Scatter(
        x=daily_row_365['Datum'],
        y=daily_row_365['KM_roll_90'],
        mode='lines',
        name='Trailing 90days km',
        line=dict(color='firebrick', width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Rolling 90d: %{y:.1f} km<extra></extra>",
    ),
    secondary_y=False
)

# 365d rolling (right y-axis)
fig_row2.add_trace(
    go.Scatter(
        x=daily_row_365['Datum'],
        y=daily_row_365['KM_roll_365'],
        mode='lines',
        name='Trailing 365days km',
        line=dict(color='rgb(34, 180, 180)', width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Rolling 365d: %{y:.1f} km<extra></extra>",
    ),
    secondary_y=True
)

# Layout (modern title API, no legacy titlefont)
fig_row2.update_layout(
    title=dict(text='Trailing 90d and 365d Rowing Volume (fig_row2)'),
    xaxis_title='Datum',
    xaxis=dict(showgrid=False),
    width=650,
    height=400,
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title_text=None,
    plot_bgcolor="white",
    hovermode='x unified',
)

# Axis titles + fonts (target axes via secondary_y)
fig_row2.update_yaxes(
    title_text='Trailing 90 Days',
    title_font=dict(color="firebrick"),
    tickfont=dict(color="firebrick"),
    showgrid=False,
    secondary_y=False
)
fig_row2.update_yaxes(
    title_text='Trailing 365 Days',
    title_font=dict(color="rgb(34, 180, 180)"),
    tickfont=dict(color="rgb(34, 180, 180)"),
    showgrid=False,
    secondary_y=True
)

# ---- Rowing Efficiency (pwr/hr) — raw daily + 28d & 90d rolling ----

df_row_eff = df_row.copy()
df_row_eff['Datum'] = pd.to_datetime(df_row_eff['Datum'], dayfirst=True, errors='coerce')
df_row_eff['pwr/hr'] = pd.to_numeric(df_row_eff['pwr/hr'], errors='coerce')

# daily mean (if multiple sessions per day)
daily_eff = (
    df_row_eff
    .dropna(subset=['Datum'])
    .set_index('Datum')
    .resample('D')['pwr/hr']
    .mean()
    .reset_index()
)

# rolling averages
daily_eff['pwr_hr_roll_28'] = daily_eff['pwr/hr'].rolling(window=28, min_periods=1).mean()
daily_eff['pwr_hr_roll_90'] = daily_eff['pwr/hr'].rolling(window=90, min_periods=1).mean()

# filter last 365 days
cutoff_365 = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
daily_eff = daily_eff[daily_eff['Datum'] >= cutoff_365]

# figure
fig_row3 = go.Figure()

# raw daily scatter
fig_row3.add_trace(go.Scatter(
    x=daily_eff['Datum'],
    y=daily_eff['pwr/hr'],
    mode='markers',
    name='Daily pwr/hr',
    marker=dict(color='rgba(34,180,180,0.7)', size=5),
    hovertemplate="Datum: %{x|%Y-%m-%d}<br>Daily pwr/hr: %{y:.2f}<extra></extra>"
))

# 28d rolling line
fig_row3.add_trace(go.Scatter(
    x=daily_eff['Datum'],
    y=daily_eff['pwr_hr_roll_28'],
    mode='lines',
    name='28d avg pwr/hr',
    line=dict(color='firebrick', width=2),
    hovertemplate="Datum: %{x|%Y-%m-%d}<br>28d avg pwr/hr: %{y:.2f}<extra></extra>"
))

# 90d rolling line
fig_row3.add_trace(go.Scatter(
    x=daily_eff['Datum'],
    y=daily_eff['pwr_hr_roll_90'],
    mode='lines',
    name='90d avg pwr/hr',
    line=dict(color='rgb(34, 180, 180)', width=2),
    hovertemplate="Datum: %{x|%Y-%m-%d}<br>90d avg pwr/hr: %{y:.2f}<extra></extra>"
))

# layout
fig_row3.update_layout(
    title="Avg of pwr/hr",
    height=400,
    width=650,
    plot_bgcolor="white",
    xaxis_title=None,
    yaxis_title=None,
    xaxis=dict(showticklabels=True, showgrid=False),
    yaxis=dict(showticklabels=True, showgrid=False),
    hovermode='x unified',
    legend=dict(orientation="h", y=1.05, x=0)  # horizontal legend above
)

# ---- Rowing Pace Benchmarks — only row20/row30/row40 (future-proof) — fig_row4 ----

ALLOWED_DURATIONS = [20.0, 30.0, 40.0]  # minutes you want to track

def _pace_to_seconds(x):
    """Return pace in seconds from hh:mm:ss, mm:ss, datetime.time, timedelta, or numeric."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float('nan')

    # pandas / numpy timedeltas
    try:
        import pandas as _pd
        import numpy as _np
        if isinstance(x, (_pd.Timedelta, _np.timedelta64)):
            return float(_pd.to_timedelta(x).total_seconds())
    except Exception:
        pass

    # datetime types
    import datetime as _dt
    if isinstance(x, _dt.time):
        return x.hour * 3600 + x.minute * 60 + x.second
    if isinstance(x, _dt.timedelta):
        return float(x.total_seconds())

    # numeric
    if isinstance(x, (int, float)) and np.isfinite(x):
        return float(x)

    # strings
    s = str(x).strip().replace(',', '.')
    if ':' in s:
        parts = s.split(':')
        try:
            if len(parts) == 2:  # mm:ss(.s)
                mm = int(parts[0]); ss = float(parts[1])
                return mm * 60 + ss
            if len(parts) == 3:  # hh:mm:ss(.s)
                hh = int(parts[0]); mm = int(parts[1]); ss = float(parts[2])
                return hh * 3600 + mm * 60 + ss
        except Exception:
            return float('nan')
    return pd.to_numeric(s, errors='coerce')

def _fmt_pace(seconds):
    if pd.isna(seconds) or not np.isfinite(seconds):
        return ''
    seconds = int(round(float(seconds)))
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"

# find the pace column robustly
def _find_pace_col(columns):
    for c in columns:
        s = str(c).strip().lower().replace(' ', '')
        if s in ('pace/500m', 'pace500m', 'paceper500m', 'pace_per_500m'):
            return c
    for c in columns:
        s = str(c).lower()
        if 'pace' in s and '500' in s:
            return c
    return 'Pace/500m'  # fallback to your original

pace_col = _find_pace_col(df_row.columns)

# --- Build steady-only subset for allowed durations
steady = df_row.copy()
steady['Datum'] = pd.to_datetime(steady['Datum'], dayfirst=True, errors='coerce')
steady['Art'] = steady['Art'].astype(str).fillna('').str.strip()

# match 'row' + digits; extract minutes
mask_row = steady['Art'].str.lower().str.replace(' ', '', regex=False).str.match(r'^row\d+$', na=False)
steady = steady[mask_row].copy()
steady['row_min'] = (
    steady['Art'].str.replace(' ', '', regex=False).str.extract(r'row(\d+)', expand=False).astype(float)
)

# keep only 20/30/40
steady = steady[steady['row_min'].isin(ALLOWED_DURATIONS)].copy()

# pace seconds
steady['pace_s'] = steady[pace_col].apply(_pace_to_seconds)
steady = steady.dropna(subset=['Datum', 'pace_s', 'row_min'])
steady = steady[np.isfinite(steady['pace_s'])].sort_values('Datum')

# 28D time-based rolling per allowed duration
roll_list = []
for dur, sub in steady.groupby('row_min', sort=True):
    g = sub.set_index('Datum').sort_index()
    g['pace_roll_28'] = g['pace_s'].rolling('28D', min_periods=1).mean()
    g['row_min'] = dur
    roll_list.append(g.reset_index())
steady_roll = pd.concat(roll_list, ignore_index=True) if roll_list else steady.iloc[0:0].copy()

# legend labels and dash styles (stable over time)
name_map = {20.0: 'row20', 30.0: 'row30', 40.0: 'row40'}
dash_cycle = {20.0: 'solid', 30.0: 'dash', 40.0: 'dot'}

# --- Figure
fig_row4 = go.Figure()

# Raw daily scatter (teal) for allowed durations only
fig_row4.add_trace(go.Scatter(
    x=steady['Datum'],
    y=steady['pace_s'],
    mode='markers',
    name='Daily pace/500m',
    marker=dict(color='rgba(34,180,180,0.7)', size=5),
    hovertemplate=(
        "Datum: %{x|%Y-%m-%d}<br>"
        "Art: %{customdata}<br>"
        "Pace/500m: %{{y:.0f}}s (%{{text}})<extra></extra>"
    ),
    customdata=steady['Art'].str.replace(' ', '', regex=False),
    text=steady['pace_s'].apply(_fmt_pace)
))

# Rolling 28d lines (firebrick) for each allowed duration
plotted = set()
for dur, sub in steady_roll.groupby('row_min', sort=True):
    label = name_map.get(dur, f"row{int(dur)}")
    fig_row4.add_trace(go.Scatter(
        x=sub['Datum'],
        y=sub['pace_roll_28'],
        mode='lines',
        name=f"28d avg {label}",
        line=dict(color='firebrick', width=2, dash=dash_cycle.get(dur, 'solid')),
        hovertemplate=(
            "Datum: %{x|%Y-%m-%d}<br>"
            f"28d avg pace ({label}): %{{y:.0f}}s (%{{text}})<extra></extra>"
        ),
        text=sub['pace_roll_28'].apply(_fmt_pace)
    ))
    plotted.add(dur)

# Add empty legend entries for durations not present yet (keeps legend stable for the future)
for dur in ALLOWED_DURATIONS:
    if dur not in plotted:
        label = name_map[dur]
        fig_row4.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name=f"28d avg {label}",
            line=dict(color='firebrick', width=2, dash=dash_cycle[dur]),
            hoverinfo='skip',
            showlegend=True
        ))

# Layout (your style)
fig_row4.update_layout(
    title="Pace/500m — row20, row30, row40 (daily & 28d avg)",
    height=400,
    width=650,
    plot_bgcolor="white",
    xaxis_title=None,
    yaxis_title=None,
    xaxis=dict(showticklabels=True, showgrid=False),
    yaxis=dict(showticklabels=True, showgrid=False),
    hovermode='x unified',
    legend=dict(orientation="h", y=1.05, x=0)
)

# Invert Y so faster = higher
fig_row4.update_yaxes(autorange="reversed")

# Pretty y ticks in mm:ss (based on available points)
if len(steady):
    ymin = int(np.nanmin(steady['pace_s']))
    ymax = int(np.nanmax(steady['pace_s']))
    lo = int(5 * round(ymin / 5))
    hi = int(5 * round(max(ymax, ymin + 10) / 5))
    ticks = np.linspace(lo, hi, 5).astype(int)
    fig_row4.update_yaxes(
        tickmode='array',
        tickvals=ticks.tolist(),
        ticktext=[_fmt_pace(t) for t in ticks.tolist()]
    )

# ---- Weekly Aerobic Load (Running + Rowing, minutes, last 365 days, Mon–Sun) — fig_row4 ----

def _to_minutes(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float('nan')
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, 'hour') and hasattr(x, 'minute'):
        sec = getattr(x, 'second', 0)
        return x.hour*60 + x.minute + sec/60
    if hasattr(x, 'total_seconds'):
        try:
            return float(x.total_seconds())/60.0
        except Exception:
            pass
    s = str(x).strip().replace(',', '.')
    if ':' in s:
        parts = s.split(':')
        try:
            if len(parts) == 2:
                m, sec = parts
                return int(m) + float(sec)/60.0
            if len(parts) == 3:
                h, m, sec = parts
                return int(h)*60 + int(m) + float(sec)/60.0
        except Exception:
            return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(s, errors='coerce')

def _week_range_str(week_start):
    ws = pd.Timestamp(week_start).normalize()
    we = ws + pd.Timedelta(days=6)
    return f"{ws:%Y-%m-%d} → {we:%Y-%m-%d}"

# === PREP: make sure inputs exist and have expected columns ===
required_run_cols = {'Datum', 'Zeit'}
required_row_cols = {'Datum'}  # Zeit or Zeit_min handled below

if not required_run_cols.issubset(df.columns):
    raise ValueError(f"df is missing columns: {required_run_cols - set(df.columns)}")
if not required_row_cols.issubset(df_row.columns):
    raise ValueError(f"df_row is missing columns: {required_row_cols - set(df_row.columns)}")

# === BUILD 'run' with parsed minutes ===
run = df.copy()
run['Datum'] = pd.to_datetime(run['Datum'], dayfirst=True, errors='coerce')
run['run_min'] = run['Zeit'].apply(_to_minutes)

# === BUILD 'row' with parsed minutes ===
row = df_row.copy()
row['Datum'] = pd.to_datetime(row['Datum'], dayfirst=True, errors='coerce')
if 'Zeit_min' in row.columns:
    row['row_min'] = pd.to_numeric(row['Zeit_min'], errors='coerce')
else:
    # falls back to 'Zeit' like running
    if 'Zeit' not in row.columns:
        raise ValueError("df_row needs either 'Zeit_min' or 'Zeit' to compute rowing minutes.")
    row['row_min'] = row['Zeit'].apply(_to_minutes)

# === WEEKLY (Mon..Sun) with correct binning (label = Monday) ===
run_week = (run.dropna(subset=['Datum'])
              .set_index('Datum')
              .assign(run_min=lambda d: pd.to_numeric(d['run_min'], errors='coerce').fillna(0.0))
              .resample('W-MON', label='left', closed='left')['run_min'].sum()
              .rename('Running')
              .to_frame())

row_week = (row.dropna(subset=['Datum'])
              .set_index('Datum')
              .assign(row_min=lambda d: pd.to_numeric(d['row_min'], errors='coerce').fillna(0.0))
              .resample('W-MON', label='left', closed='left')['row_min'].sum()
              .rename('Rowing')
              .to_frame())

# --- Combine and restrict to last 365 days ---
both = run_week.join(row_week, how='outer').fillna(0.0)

today = pd.Timestamp.today().normalize()
latest_monday = today - pd.Timedelta(days=today.weekday())  # Monday of this week

end_wk = latest_monday
start_wk = (end_wk - pd.Timedelta(days=364)).normalize()

grid_start = start_wk  # already a Monday after resample
all_weeks = pd.date_range(start=grid_start, end=end_wk, freq='W-MON')

wide = (pd.DataFrame(index=all_weeks)
        .join(both, how='left')
        .fillna(0.0)
        .rename_axis('week_start')
        .reset_index())

wide['Total'] = wide['Running'] + wide['Rowing']
wide['WeekLabel'] = wide['week_start'].apply(_week_range_str)

# --- Figure ---
fig_row5 = go.Figure()

# Running = firebrick
fig_row5.add_trace(go.Bar(
    x=wide['week_start'],
    y=wide['Running'],
    name='Running',
    marker=dict(color='firebrick'),
    customdata=np.stack([wide['WeekLabel'], wide['Total']], axis=1),
    hovertemplate=("Week: %{customdata[0]}<br>"
                   "Running: %{y:.0f} min<br>"
                   "Total: %{customdata[1]:.0f} min<extra></extra>")
))

# Rowing = teal
fig_row5.add_trace(go.Bar(
    x=wide['week_start'],
    y=wide['Rowing'],
    name='Rowing',
    marker=dict(color='rgb(34,180,180)'),
    customdata=np.stack([wide['WeekLabel'], wide['Total']], axis=1),
    hovertemplate=("Week: %{customdata[0]}<br>"
                   "Rowing: %{y:.0f} min<br>"
                   "Total: %{customdata[1]:.0f} min<extra></extra>")
))

fig_row5.update_layout(
    title="Weekly Aerobic Load (Running + Rowing, minutes, last 365 days)",
    height=400,
    width=650,
    barmode='stack',
    plot_bgcolor="white",
    xaxis_title=None,
    yaxis_title=None,
    xaxis=dict(showticklabels=True, showgrid=False),
    yaxis=dict(showticklabels=True, showgrid=False),
    hovermode='x unified',
    legend=dict(orientation="h", y=1.05, x=0)
)

###
# Rowing Best Times (exact test distances + PB delta)
###

# --- 1) Working copy ---
row_bt = df_row.copy()

# --- 2) Normalize fields ---
row_bt['Datum'] = pd.to_datetime(row_bt['Datum'], dayfirst=True, errors='coerce')
row_bt['Zeit_min'] = pd.to_numeric(row_bt['Zeit_min'], errors='coerce')

row_bt['art_norm'] = (
    row_bt['Art']
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(' ', '', regex=False)
)

# --- 3) Valid test workouts and exact distances ---
VALID_TESTS = {
    'row1k': 1000,
    'row2k': 2000,
    'row5k': 5000,
    'row10k': 10000,
    'rowhw': 21097.5,
    'rowm': 42195,
}

DIST_ORDER = [
    'row1k',
    'row2k',
    'row5k',
    'row10k',
    'rowhw',
    'rowm',
]

LABELS = {
    'row1k': '1 km',
    'row2k': '2 km',
    'row5k': '5 km',
    'row10k': '10 km',
    'rowhw': 'Half Marathon',
    'rowm': 'Marathon',
}

# keep only exact test workouts
row_bt = row_bt[row_bt['art_norm'].isin(VALID_TESTS)].copy()

# --- 4) Map target distance + pace ---
row_bt['target_m'] = row_bt['art_norm'].map(VALID_TESTS)
row_bt['pace_s'] = row_bt['Zeit_min'] * 60 / (row_bt['target_m'] / 500)

row_bt = row_bt.dropna(subset=['Datum', 'Zeit_min', 'pace_s'])

# --- 5) Compute PB + delta vs previous PB ---
best_rows = []

for art in DIST_ORDER:
    sub = row_bt[row_bt['art_norm'] == art].sort_values('Datum')
    if sub.empty:
        continue

    # best (fastest)
    best = sub.loc[sub['pace_s'].idxmin()]

    # previous best BEFORE this PB
    before = sub[sub['Datum'] < best['Datum']]
    if not before.empty:
        prev_best = before['pace_s'].min()
        delta_s = best['pace_s'] - prev_best   # negative = improvement
    else:
        delta_s = None

    best_rows.append({
        'Distanz': LABELS[art],
        'Zeit': best['Zeit'],
        'Pace /500m': _fmt_pace(best['pace_s']),
        'Δ vs prev (s)': f"{int(round(delta_s)):+d}" if delta_s is not None else '',
        'Datum': best['Datum'].strftime('%Y-%m-%d'),
        'Workout': best['Art'],
        'Watt': round(best['Watt'], 0) if 'Watt' in best and pd.notna(best['Watt']) else None,
        '_order': DIST_ORDER.index(art),
        '_delta_raw': delta_s,
    })

# --- 6) Final table (sorted) ---
best_table = (
    pd.DataFrame(best_rows)
    .sort_values('_order')
    .drop(columns=['_order'])
)

# --- 7) Dash DataTable ---
table_best = dash_table.DataTable(
    data=best_table.drop(columns=['_delta_raw']).to_dict("records"),
    columns=[{"name": c, "id": c} for c in best_table.columns if c != '_delta_raw'],
    style_cell={
        "textAlign": "center",
        "padding": "6px",
        "fontFamily": "monospace",
        "fontSize": "13px",
        "border": "none",
    },
    style_header={
        "fontWeight": "600",
        "borderBottom": "1px solid #ddd",
        "backgroundColor": "white",
    },
    style_data_conditional=[
        # alternating rows
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "#fafafa",
        },
        # --- Highlight 2k ---
        {
        "if": {"filter_query": "{Distanz} = '2 km'"},
        "backgroundColor": "rgba(178,34,34,0.08)",
        "borderLeft": "4px solid firebrick",
        "fontWeight": "600",
    },
        # PB improvement (negative delta = faster)
        {
            "if": {
                "filter_query": "{Δ vs prev (s)} contains '-'",
                "column_id": "Δ vs prev (s)",
            },
            "color": "firebrick",
            "fontWeight": "600",
        },
    ],
)

def derived_from_2k(pace_2k):
    return {
        "5k Pace": (_fmt_pace(pace_2k + 20), _fmt_pace(pace_2k + 22)),
        "10k Pace": (_fmt_pace(pace_2k + 25), _fmt_pace(pace_2k + 30)),
        "Steady": (_fmt_pace(pace_2k + 35), _fmt_pace(pace_2k + 45)),
        "Threshold": (_fmt_pace(pace_2k + 20), _fmt_pace(pace_2k + 25)),
        "VO2max": (_fmt_pace(pace_2k + 5), _fmt_pace(pace_2k + 10)),
    }

def age_class_2k_label(pace_s):
    total_2k = pace_s * 4  # Sekunden
    if total_2k < 420: return "Elite"
    if total_2k < 450: return "Sehr gut"
    if total_2k < 480: return "Gut"
    if total_2k < 510: return "Trainiert"
    return "Freizeit"

row2k = row_bt[row_bt['art_norm'] == 'row2k']

best_2k = None
derived = None

if not row2k.empty:
    best_2k = row2k.loc[row2k['pace_s'].idxmin()]
    derived = derived_from_2k(best_2k['pace_s'])

two_k_card = None

if best_2k is not None:
    two_k_card = html.Div(
        children=[
            html.H4("2k Summary (Reference)", style={'marginBottom': '12px'}),

            html.P(f"Age Group: Men 40–44"),
            html.P(f"2k Pace: {_fmt_pace(best_2k['pace_s'])} /500m"),
            html.P(f"Classification: {age_class_2k_label(best_2k['pace_s'])}"),

            html.Hr(),

            html.P("Derived Targets:", style={'fontWeight': '600'}),
            html.Ul([
                html.Li(f"5k Pace: {derived['5k Pace'][0]} – {derived['5k Pace'][1]}"),
                html.Li(f"10k Pace: {derived['10k Pace'][0]} – {derived['10k Pace'][1]}"),
                html.Li(f"Steady: {derived['Steady'][0]} – {derived['Steady'][1]}"),
                html.Li(f"Threshold: {derived['Threshold'][0]} – {derived['Threshold'][1]}"),
                html.Li(f"VO₂max: {derived['VO2max'][0]} – {derived['VO2max'][1]}"),
            ]),
        ],
        style={
            'margin': '20px',
            'padding': '16px 20px 24px 20px',
            'backgroundColor': 'white',
            'border': '1px solid #e6e6e6',
            'borderRadius': '6px',
            'width': '650px',
        }
    )



# Make changes to all figures
_ALL_FIGS = [
    fig1, fig2, fig3, fig5, fig6, fig8, fig9, fig11, fig13, fig14, fig15, fig16, fig17, fig18, fig_row2, fig_row3, fig_row4, fig_row5
]

for _f in _ALL_FIGS:
    _f.update_layout(
        xaxis_title=None,
        width=650,  # Set the width of the graph
        height=400  # Set the height of the graph
    )

# DASHBOARD

app.layout = html.Div([
    dcc.Tabs([
# Tab 1 - Wertetabelle        
dcc.Tab(label='Important Values', children=[
    html.Div(
        children=[
            dcc.Graph(figure=fig23, config={'displayModeBar': False})
        ],
        style={'display':'inline-block', 'width':'1200px', 'padding-top':'10px', 'padding-bottom':'10px', 'padding-left':'20px', 'margin': '10px', 'margin-top': '20px'},
    ),
        
    html.Div(
    children=[    
        html.P(children=[
            html.Span(html.B("Today's run: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_workout}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Distance:"), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_km} km", style={'display': 'inline-block', 'width': '150px'}),
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(html.B("Duration: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_time}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Avg Pace: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{formatted_todays_pace}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(html.B("Avg Power: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_watt} W", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Avg HR: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_HFQ}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(html.B("Pwr/Hr: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_pwr_hr}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Zone (W)"), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_zone_start} - {todays_zone_end}", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),        
    ],
    style={'display':'inline-block','width':'590px','border': '1px solid black', 'padding-left': '10px', 'padding-right': '10px', 'margin': '10px', 'background-color': 'rgba(178, 34, 34, 0.4)'}  # Adjust width as needed
),

    html.Div(
    children=[    
        html.P(children=[
            html.Span(html.B("Kilometers YTD: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{YTD_km_data} km", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Elevation gain YTD: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{YTD_HM_data} m", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(html.B(f"{current_year} goal: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{running_goal} km", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("KM with Kids: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_sum_kids} km", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(html.B(f"Δ to {current_year} goal: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_goal_difference} km", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B(f"KM with Kids {current_year}: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_ytd_km_sum_kids_j} km", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(html.B("Today's CP:"), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_CP_value} W  ({max_CP} W)", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Peaks:"), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{peaks_completed} / {peaks_total} - {peaks_percentage}%", style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
    ],
    style={'display':'inline-block','width': '590px', 'border': '1px solid black', 'padding-left': '10px', 'padding-right': '10px', 'margin': '10px', 'background-color': 'rgba(64, 162, 111, 0.5)'}  # Adjust width as needed
),

    html.Div(
        children=[    
            html.P(children=[
                html.Span(html.B("Tomorrow's run: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_workout}", style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("Distance:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_km} km", style={'display': 'inline-block', 'width': '150px'}),
            ], style={'display': 'flex', 'align-items': 'baseline'}),

            html.P(children=[
                html.Span(html.B("Duration: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_time}", style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("Zone (W)"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{tomorrows_zone_start} - {tomorrows_zone_end}", style={'display': 'inline-block', 'width': '150px'})
          ], style={'display': 'flex', 'align-items': 'baseline'}),  
        ],
        style={'display':'inline-block','width': '590px','border': '1px solid black', 'padding-left': '10px', 'padding-right': '10px', 'margin': '10px', 'background-color': 'rgba(153, 97, 0, 0.5)'}
        ),

    html.Div(
    children=[    
        html.P(children=[
            html.Span(html.B("Today's CTL: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{rounded_rCTL_value}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Pwr/Hr (ø 42 d): "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span([f"{rounded_pwr_hr_42} ", html.Span(pwr_hr_arrow, style={'fontSize': '0.8em'}), f" ({max_pwr_hr})"], style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),

        html.P(children=[
            html.Span(html.B("Today's ATL: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span(f"{todays_ATL}", style={'display': 'inline-block', 'width': '150px'}),
            html.Span(html.B("Today's load: "), style={'display': 'inline-block', 'width': '200px'}),
            html.Span([f"{todays_load} ", html.Span(load_arrow, style={'fontSize': '0.8em'})], style={'display': 'inline-block', 'width': '150px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
    ],
    style={'display':'inline-block','width': '590px', 'border': '1px solid black', 'padding-left': '10px', 'padding-right': '10px', 'margin': '10px', 'background-color': 'rgba(113, 135, 38, 0.5)'}  # Adjust width as needed
),

html.Div(
    style={'display': 'flex', 'align-items': 'flex-start'},
    children=[
    html.Div(children=[
    html.Div(
        children=[    
            html.P(children=[
                html.Span(html.B("5k: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{time_5k} ({CP_5k} W)", style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("10k: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{time_10k} ({CP_10k} W)", style={'display': 'inline-block', 'width': '150px'})
            ], style={'display': 'flex', 'align-items': 'baseline'}),

            html.P(children=[
                html.Span(html.B("Half Marathon: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{time_HM} ({CP_HM} W)", style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("Marathon: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span(f"{time_marathon} ({CP_M} W)", style={'display': 'inline-block', 'width': '150px'})
            ], style={'display': 'flex', 'align-items': 'baseline'}),
        ],
        style={'width':'590px','border': '1px solid black', 'padding-left': '10px', 'padding-right': '10px', 'margin':'10px', 'background-color': 'rgba(34, 180, 180, 0.5)'}  # Adjust width as needed
),      

    html.Div(
        children=[    
            html.P(children=[
                html.Span(html.B("Last Week's KM: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{last_weeks_KM} ", html.Span(weekly_arrow_KM, style={'fontSize': '0.8em'})], style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("Last Week's RSS: "), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{last_weeks_RSS} ", html.Span(weekly_arrow_RSS, style={'fontSize': '0.8em'})], style={'display': 'inline-block', 'width': '150px'})
            ], style={'display': 'flex', 'align-items': 'baseline'}),
            
            html.P(html.Span(), style={'borderBottom': '1px solid black'}),

            html.P(children=[
                html.Span(html.B("Last 7d KM:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{last_7d_KM_rounded} ", html.Span(arrow_7d_KM, style={'fontSize': '0.8em'})], style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("Last 7d RSS:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{last_7d_RSS} ", html.Span(arrow_7d_RSS, style={'fontSize': '0.8em'})], style={'display': 'inline-block', 'width': '150px'}),
            ], style={'display': 'flex', 'align-items': 'baseline'}),

            html.P(children=[
                html.Span(html.B("Last 7d time:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{last_7d_time}"], style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("Longrun ratio:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{percent_longrun} %"], style={'display': 'inline-block', 'width': '150px'})
            ], style={'display': 'flex', 'align-items': 'baseline'}),            
            
            html.P(html.Span(), style={'borderBottom': '1px solid black'}),

            html.P(children=[
                html.Span(html.B("90d running volume:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{cumulative_sum_90_today} km"], style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("365d running volume:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{cumulative_sum_365_today} km"], style={'display': 'inline-block', 'width': '150px'})
            ], style={'display': 'flex', 'align-items': 'baseline'}),

            html.P(children=[
                html.Span(html.B("90d rowing volume:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{row_vol_90} km"], style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("365d rowing volume:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{row_vol_365} km"], style={'display': 'inline-block', 'width': '150px'})
            ], style={'display': 'flex', 'align-items': 'baseline'}),       

        ],
        style={'width':'590px','border': '1px solid black', 'padding-left': '10px', 'padding-right': '10px', 'margin':'10px', 'background-color': 'rgba(178, 34, 34, 0.4)'}  # Adjust width as needed
),],
),

 html.Div(
        children=[    
        dash_table.DataTable(
            id='running-km-table',
            columns=[
            {'name': 'Schuh', 'id': 'Schuh'},
            {'name': 'Total Running KM', 'id': 'KM'},
        ],
        data=df_Schuhe,
        style_table={
            'width': '99%',
            'border': 'none',
            'margin': '5px',
        },
        style_header={
            'backgroundColor': 'rgba(34, 180, 180, 0.5)',
            'color': 'black',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'left',
            'border': 'none',
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'KM', 'filter_query': '{KM} > 800'},
                'backgroundColor': 'red',
                'color': 'white',
            },
            {
                'if': {'column_id': 'KM', 'filter_query': '{KM} > 500 and {KM} <= 800'},
                'backgroundColor': 'yellow',
            },
        ],),
        ],    
        style={'width':'590px','display':'inline-block', 'align-items': 'flex-start', 'margin': '10px', 'padding-left': '10px', 'padding-right': '10px'}  # Adjust width as needed
),]),
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
            html.Div(dcc.Graph(figure=fig9, config={'displayModeBar': False})),
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

# Tab 4 - Daten current year
dcc.Tab(label=f'{current_year}', children=[
        html.Div([
        # Goal Graph for current year
        html.Div(dcc.Graph(figure=fig15, config={'displayModeBar': False})),
        html.Div(dcc.Graph(figure=fig18, config={'displayModeBar': False})),
    ], style={'display': 'flex'}),     
        html.Div([
        # Monthly Km for current year
        html.Div(dcc.Graph(figure=fig17, config={'displayModeBar': False})),
        # weekly Km and TSS for current year
        html.Div(dcc.Graph(figure=fig14, config={'displayModeBar': False})),
    ], style={'display': 'flex'}),  
        ]),

# Tab 5 - Rowing Values
dcc.Tab(
    label='Rowing Figures',
    children=[

        # --- Calendar heatmap ---
        html.Div(
            children=[
                dcc.Graph(
                    figure=fig_row1,
                    config={'displayModeBar': False}
                )
            ],
            style={
                'display': 'inline-block',
                'width': '1200px',
                'paddingTop': '10px',
                'paddingBottom': '10px',
                'paddingLeft': '20px',
                'margin': '10px',
                'marginTop': '20px',
            },
        ),

        # --- Row 1: rolling volume + efficiency ---
        html.Div(
            children=[
                html.Div(
                    dcc.Graph(
                        figure=fig_row2,
                        config={'displayModeBar': False}
                    )
                ),
                html.Div(
                    dcc.Graph(
                        figure=fig_row3,
                        config={'displayModeBar': False}
                    )
                ),
            ],
            style={'display': 'flex'},
        ),

        # --- Row 2: pace benchmarks + weekly load ---
        html.Div(
            children=[
                html.Div(
                    dcc.Graph(
                        figure=fig_row4,
                        config={'displayModeBar': False}
                    )
                ),
                html.Div(
                    dcc.Graph(
                        figure=fig_row5,
                        config={'displayModeBar': False}
                    )
                ),
            ],
            style={'display': 'flex'},
        ),

        # --- Best distances table ---
        html.Div(
            children=[
                html.H4(
                    "Rowing Best Distances",
                    style={'marginBottom': '12px'},
                ),
                table_best,
            ],
            style={
                # outer spacing
                'margin': '20px',
                'marginBottom': '80px',

                # inner spacing (prevents edge sticking)
                'padding': '16px 20px 24px 20px',

                # card look
                'backgroundColor': 'white',
                'border': '1px solid #e6e6e6',
                'borderRadius': '6px',

                # avoid visual clipping at bottom
                'minHeight': 'fit-content',
            },
        ),

        # --- 2k summary card ---
        html.Div(
            children=[two_k_card],
            style={'display': 'flex'},
        ),

    ],
),


# Tab 6 - Peaks Map
        dcc.Tab(label='Peaks Map', children=[
            html.Div([
                html.Iframe(
                    srcDoc=open('../peaks_projekt/Peaks_Map/peaks_progress.html', 'r').read(),
                    width='1200px',
                    height='650px',
                    style={'margin': '20px', 'border':'1px'},
                ),
            ]),
            html.Div(
    children=[    
        dash_table.DataTable(
            id='peaks-table',
            columns=[
                {'name': '#', 'id': 'index'},
                {'name': 'Peak', 'id': 'name'},
                {'name': 'Elevation', 'id': 'elevation'},
                {'name': 'Completed', 'id': 'gelaufen'}
            ],
            data=df_peaks.reset_index().to_dict('records'),
            style_table={
            'width': '50%',
            'border': 'none',
            'margin': '50px',
        },             
            style_header={
            'backgroundColor': 'rgba(178, 34, 34, 0.5)',
            'color': 'black',
            'fontWeight': 'bold'
        },)
            ])
        ]),
    ]),
])    

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)