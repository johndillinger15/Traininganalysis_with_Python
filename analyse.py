import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load data from Excel file
df = pd.read_excel("training_data.xlsx", sheet_name="2018+", usecols="C:O,S:AB")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Datum'])

# Extract year and month from the Date column
df['Year'] = df['Date'].dt.year

# Filter data for the current year and up to today
current_year = pd.to_datetime('today').year
df_ytd = df[(df['Year'] == current_year) & (df['Date'] <= pd.to_datetime('today'))].copy()

# Format the 'Date' column in German time format
df_ytd['Date'] = df_ytd['Date'].dt.strftime('%d.%m.%Y')

# Plotting the first line chart (tCTL vs rCTL) using Plotly Express
fig1 = px.line(df_ytd, x='Date', y=['tCTL', 'rCTL'], title=f'YTD Comparison of CTL for {current_year}',
              labels={'value': 'CTL Value', 'variable': 'Metric'},
              line_shape='linear', render_mode='svg')

# Plotting the second line chart (TSS vs RSS) using Plotly Express
fig2 = px.line(df_ytd, x='Date', y=['TSS load', 'RSS load'], title=f'YTD Comparison of load for {current_year}',
              labels={'value': 'load Value', 'variable': 'Metric'},
              line_shape='linear', render_mode='svg')
fig2.add_hline(y=0.8, line_width=3, line_dash="solid", line_color="lightgreen")
fig2.add_hline(y=1, line_width=3, line_dash="solid", line_color="green")
fig2.add_hline(y=1.3, line_width=3, line_dash="solid", line_color="orange")
fig2.add_hline(y=1.5, line_width=3, line_dash="solid", line_color="red")


# Show the plot
fig1.show()
fig2.show()