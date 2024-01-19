from dash import Dash, html, dcc
import pandas as pd
import plotly.express as px

# Load app
app = Dash(__name__)

# Load data from Excel file
df = pd.read_excel("training_data.xlsx", sheet_name="2018+", usecols="C:O,S:AD")

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
              labels={'value': 'CTL', 'variable': 'Metric'},
              line_shape='linear', 
              render_mode='svg', 
              template='plotly')

# fig1.update_layout(showlegend=False)

# Plotting the second line chart (TSS vs RSS) using Plotly Express
fig2 = px.line(df_ytd, x='Date', y=['TSS load', 'RSS load'], title=f'YTD Comparison of load for {current_year}',
              labels={'value': 'load', 'variable': 'Metric'},
              line_shape='linear', render_mode='svg')
fig2.add_hline(y=0.8, line_width=3, line_dash="solid", line_color="lightgreen")
fig2.add_hline(y=1, line_width=3, line_dash="solid", line_color="green")
fig2.add_hline(y=1.3, line_width=3, line_dash="solid", line_color="orange")
fig2.add_hline(y=1.5, line_width=3, line_dash="solid", line_color="red")

# Plotting the third line chart FTP vs CP using Plotly Express
fig3 = px.line(df_ytd, x='Date', y=['FTP', 'CP'], title=f'YTD Comparison of CP vs FTP for {current_year}',
              labels={'value': 'CP/FTP', 'variable': 'Metric'},
              line_shape='linear', render_mode='svg')

# Dashboard
app.layout = html.Div(children=[
    html.H1(children='Testdashboard',
            style={'textAlign': 'center'}),

    html.H3(children="RSS vs TSS Vergleich"),

    # Use a single row for the three figures
    html.Div([
        # First figure
        html.Div(dcc.Graph(figure=fig1)),
        # Second figure
        html.Div(dcc.Graph(figure=fig2)),
        # Third figure
        html.Div(dcc.Graph(figure=fig3)),
    ], style={'display': 'flex'}),

])

if __name__ == '__main__':
    app.run(debug=True)
