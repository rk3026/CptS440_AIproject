import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Sample Data
df = px.data.iris()

# Create a Plotly Figure
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

# Initialize Dash App
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("My First Plotly Dash App"),
    dcc.Graph(figure=fig)
])

# Run the App
if __name__ == '__main__':
    app.run_server(debug=True)
