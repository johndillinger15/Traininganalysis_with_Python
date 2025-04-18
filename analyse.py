from dash import Dash, html, dcc, dash_table
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
df = pd.read_excel("../training.xlsx", sheet_name="2018+", usecols="C:Y")
df_active_shoes = pd.read_excel('../training.xlsx' , sheet_name="Schuhe", usecols="A")
df_active_shoes.dropna(inplace=True)
df_peaks = pd.read_csv("../peaks_projekt/Peaks_Map/peaks_data.csv")
df_raw_peaks = pd.read_csv("../peaks_projekt/Peaks_Map/peaks_raw_data.csv")
df_peaks = df_peaks.filter(['name','elevation','gelaufen'])
df_peaks.index = df_peaks.index +1

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
df_last_7_days = df[(df['Date'] >= seven_days_ago) & (df['Date'] <= today2)]# Convert 'Zeit' column to timedelta
# df_last_7_days['Zeit'] = pd.to_timedelta(df_last_7_days['Zeit'].astype(str))
df_last_7_days.loc[:, 'Zeit'] = pd.to_timedelta(df_last_7_days['Zeit'].astype(str))
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
rounded_ytd_km_sum_kids_j = int(round(ytd_km_sum_kids_j['KM'].iloc[-1],0))

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
fig3 = px.line(df_last_365_days, x='Date', y=['CP'], title='CP for last 365 days (fig3)',
               labels={'value': 'CP/FTP', 'variable': 'Metric'},
               line_shape='linear', color_discrete_sequence=['#2283B4'])

# Update layout of the figure for primary axis
fig3.update_layout(
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
    title='Yearly Running Volume (fig5)',
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
# Create a figure with two subplots
fig8 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for Cumulative_Sum_90_Days on the left y-axis
fig8.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_90_Days'], mode='lines', name='Trailing 90days km', line=dict(color='firebrick')),)

# Add traces for Cumulative_Sum_365_Days on the right y-axis
fig8.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cumulative_Sum_365_Days'], mode='lines', name='Trailing 365days km', line=dict(color='rgb(34, 180, 180)')), secondary_y=True)

# Update layout with titles and labels
fig8.update_layout(
    title='Trailing 90d and 365d Running Volume (fig8)',
    xaxis_title='Date',
    yaxis_title='Trailing 90 Days',
    yaxis2_title='Trailing 365 Days',
    yaxis2=dict(titlefont=dict(color="#2283B4"), tickfont=dict(color="#2283B4"),),
    yaxis=dict(titlefont=dict(color="firebrick"), tickfont=dict(color="firebrick")),
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    plot_bgcolor="white",
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
# Create a Plotly figure
fig11 = px.bar(weekly_data, x='Date', y='KM', labels={'KM': 'KM'}, color_discrete_sequence=['rgb(34, 180, 180)'])

# Add a bar trace for 'RSS' on the secondary y-axis
fig11.add_trace(px.scatter(weekly_data, x='Date', y='RSS', labels={'RSS': 'RSS'}, color_discrete_sequence=['firebrick']).update_traces(yaxis='y2').data[0])

ymax1 = weekly_data['RSS'].max()*1.10
fig11.update_layout(
    title='Weekly KM vs RSS (fig11)',
    yaxis=dict(range=[0, ymax1/5], titlefont=dict(color="#2283B4"), tickfont=dict(color="#2283B4"),),
    yaxis2=dict(title='RSS', overlaying='y', side='right', range=[0, ymax1],titlefont=dict(color="firebrick"), tickfont=dict(color="firebrick")),
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white"
)
fig11.add_hline(y=30, line=dict(color="black", width=1, dash="dash"))
fig11.add_hline(y=40, line=dict(color="black", width=1, dash="dash"))

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
    title=f'Goal vs actual {current_year} (fig15)',
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
# Create a Plotly figure
fig17 = px.bar(weekly_data_current_year, x='Date', y='KM', labels={'KM': 'KM'}, color_discrete_sequence=['rgb(34, 180, 180)'])

# Add a bar trace for 'RSS' on the secondary y-axis
fig17.add_trace(px.scatter(weekly_data_current_year, x='Date', y='RSS', labels={'RSS': 'RSS'}, color_discrete_sequence=['firebrick']).update_traces(yaxis='y2').data[0])

ymax2 = weekly_data_current_year['RSS'].max()*1.10
fig17.update_layout(
    title=f'Weekly KM vs RSS for {current_year} (fig17)',
    yaxis=dict(range=[0, ymax2/5], titlefont=dict(color="#2283B4"), tickfont=dict(color="#2283B4")),
    yaxis2=dict(title='RSS', overlaying='y', side='right', range=[0, ymax2], titlefont=dict(color="firebrick"), tickfont=dict(color="firebrick")),
    width=650,  # Set the width of the graph
    height=400,  # Set the height of the graph
    legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    legend_title=None,
    plot_bgcolor="white"
)
fig17.add_vline(x=f"{today}", line_width=3, line_dash="dash", line_color="rgb(153, 97, 0)")
fig17.add_hline(y=30, line=dict(color="black", width=1, dash="dash"))
fig17.add_hline(y=40, line=dict(color="black", width=1, dash="dash"))

# This year's load
fig18 = px.line(df_current_year, x='Date', y=['load'],
              labels={'value': 'load', 'variable': 'load'},
              line_shape='linear', color_discrete_sequence=['firebrick'])

fig18.update_layout(
    title=f'Load comparison of {current_year} (fig18)',
    yaxis=dict(range=[0,2]),
    width=650,  # Set the width of the graph
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


# Activity Graph
df_last_90_days.fillna({'KM':0, 'RSS':0, 'HFQ':0, 'W1':0, 'W2':0, 'HM':0, 'Pace':0}, inplace=True)
fig23 = px.scatter(df_last_90_days, x='Date', y='KM', size='RSS', hover_data=['Art','Date','KM','RSS'])
first_of_month = df_last_90_days[df_last_90_days['Date'].dt.day == 1]['Date'].tolist()
fig23.update_layout(
    height=75,  # Set the height of the graph
    width=1240,
    plot_bgcolor="white",
    xaxis_title=None,
    yaxis_title=None,
    xaxis=dict(showticklabels=False, gridcolor='rgb(34, 180, 180)', gridwidth=2, tickvals=first_of_month),
    yaxis=dict(showticklabels=False, showgrid=False, range=[0, None]),
    margin=dict(l=0, r=0, b=0, t=0)
)
fig23.update_traces(marker_color='firebrick')


# DASHBOARD

app.layout = html.Div([
    dcc.Tabs([
# Tab 1 - Wertetabelle        
dcc.Tab(label='Important Values', children=[
    html.Div(
        children=[
            dcc.Graph(figure=fig23, config={'displayModeBar': False})
        ],
        style={'display':'inline-block', 'width':'1240px', 'padding-top':'10px', 'padding-bottom':'10px', 'margin': '10px', 'margin-top': '20px'},
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
                html.Span([f"{cumulative_sum_90_today}"], style={'display': 'inline-block', 'width': '150px'}),
                html.Span(html.B("365d running volume:"), style={'display': 'inline-block', 'width': '200px'}),
                html.Span([f"{cumulative_sum_365_today}"], style={'display': 'inline-block', 'width': '150px'})
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

# Tab 5 - Peaks Map
        dcc.Tab(label='Peaks Map', children=[
            html.Div([
                html.Iframe(
                    srcDoc=open('../peaks_projekt/Peaks_Map/peaks_progress.html', 'r').read(),
                    width='100%',
                    height='650px',
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
    app.run(debug=True)

#if __name__ == "__main__":
#    app.run_server(debug=True, host='0.0.0.0', port=8050)