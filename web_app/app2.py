from dash.dependencies import Input, Output
from transformers import pipeline
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import csv
import os

from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px  # Assuming you're using Plotly for charts



# Define sentiment analysis models https://huggingface.co/docs/transformers/main_classes/pipelines 
models = {
    "Twitter RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
    "Yelp BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
    #"xlnet-base-cased": pipeline("sentiment-analysis", model="xlnet-base-cased"),
    #"t5-small": pipeline("text2text-generation", model="t5-small")
    "llama3": "x",
    "deepseek": "y",
    "GPT-90": "z"
    
}

# Sample Data (Replace these with your actual data)
sample_data = {
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Value': [10, 20, 30, 40, 50],
    'Score': [5, 15, 25, 35, 45]
}
df = pd.DataFrame(sample_data)

# Create a simple figure (replace this with your actual visualizations)
fig1 = px.bar(df, x='Category', y='Value', title="Bar Chart Example")
fig2 = px.line(df, x='Category', y='Score', title="Line Chart Example")
fig3 = px.pie(df, names='Category', values='Value', title="Pie Chart Example")
fig4 = px.scatter(df, x='Category', y='Score', title="Scatter Plot Example")


# Metrics Tab figures
fig5 = px.ecdf(df, x='Category', y='Value', title='train vs test')
fig6 = px.ecdf(df, x='Category', y='Value', title='train vs test')
fig7 = px.ecdf(df, x='Category', y='Value', title='train vs test')
fig8 = fig = px.density_heatmap(df, x="Value", y="Score", text_auto=True)


# Other Tab Figures


# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='Multi-Chart Dashboard')
server = app.server

# Define the layout of the app
app.layout = html.Div([
    dbc.Container([
        
        # Different Tabs
        dbc.Row([
            dbc.Col(html.Img(src="./assets/logo.png", width=150), width=2),
            dbc.Col(
                dcc.Tabs(id='tabs', value='overview', children=[
                    dcc.Tab(label='Overview', value='overview'),
                    dcc.Tab(label='Metics', value='metrics'),
                    dcc.Tab(label='Other', value='other')
                ], style={'marginTop': '15px', 'width': '600px', 'height': '50px'})
            , width=6),
        ]),

        # Text input for user to enter text
        dbc.Row([
            
            dcc.Textarea(
                id="input-text",
                value="Enter text for sentiment analysis",
                style={'marginTop': '15px','width': '50%', 'height': 100}
            ),
            
            # Checkboxes for selecting models
            dcc.Checklist(
                id="model-selection",
                options=[{'label': model, 'value': model} for model in models.keys()],
                value=[" Twitter RoBERTa", " Yelp BERT"," llama3"," deepseek"," GPT-90"],  # Default models
                style={'marginTop': '20px',
                       'display': 'flex',  # Use flexbox layout
                       'flex-wrap': 'wrap',  # Allow the items to wrap to the next line
                       'gap': '20px',  # Add space between the checkboxes
                       'width': '300px'  # Set a width to control the number of items in each row (3 items in this case)
                       }
            )
        ]),
        
        dbc.Row([
            # Button to trigger analysis
            html.Button("Analyze", id="analyze-btn", n_clicks=0),
            
            # Dropdown to select previous test case results
            dcc.Dropdown(
                id="previous-results-dropdown",
                options=[{'label': f"Test Case Set {i+1}", 'value': i+1} for i in range(2)],
                placeholder="Select previous test case results"
            ),

        ]),

        # Scrollable DataTable with top margin
        dbc.Row([  
            dash_table.DataTable(
                data=df.to_dict('records'),
                page_size=10,
                style_table={
                    'marginTop': '20px',  # Add top margin
                    'height': '300px',  # Set a fixed height
                    'overflowY': 'auto',  # Enable vertical scrolling
                    'width': '100%'  # Ensures it takes up full width
                }
            ),
        ]),



        dbc.Row([  
            dcc.Loading([ 
                html.Div(id='tabs-content')
            ], type='default', color='#deb522')
        ])
    ])
])

# Define the callback to switch content based on the selected tab
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
    #Output("results-plot", "figure"),
    #Output("previous-results-dropdown", "options"),
    #Input("analyze-btn", "n_clicks"),
    #Input("input-text", "value"),
    #Input("model-selection", "value"),
    #Input("previous-results-dropdown", "value")
)
def update_content(tab):
    if tab == 'overview':
        return html.Div([
            html.Div([
                dcc.Graph(id='graph1', figure=fig1),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='graph2', figure=fig2),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='graph3', figure=fig3),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='graph4', figure=fig4),
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    elif tab == 'metrics':
        # Replace with your actual visualization functions for the 'content' tab
       return html.Div([
            html.Div([
                dcc.Graph(id='Test V.S. Train', figure=fig5),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='F1 Score', figure=fig6),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='Accuracy', figure=fig7),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='Confusion Matrix', figure=fig8),
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    elif tab == 'other':
        # Replace with your actual visualization functions for the 'year' tab
        return html.Div([
            dcc.Graph(id='graph1', figure=fig2),
        ])

# Run the server
if __name__ == "__main__":
    app.run(debug=True, port=4200)
