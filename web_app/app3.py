import os
import json
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from callbacks import register_callbacks

# Define Dash app and layout
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='Sentiment Analysis Dashboard')
server = app.server

app.layout = dbc.Container([  
    dbc.Row([  
        dbc.Col(html.H2("Sentiment Analysis Dashboard", className="text-center my-auto"), width=True),
    ], align="center", className="my-3"),    

    dbc.Row([  
        dbc.Col([  
            dcc.Tabs(id="tabs", children=[  
                dcc.Tab(label='Text Analysis', children=[  
                    dbc.Card([   
                        dbc.CardHeader("Enter Text for Analysis"),  
                        dbc.CardBody([  
                            dcc.Textarea( id="input-text", value="Enter text here", style={'width': '100%', 'height': 100} ),  
                            dbc.Button("Analyze", id="analyze-btn", color="primary", className="my-2")  
                        ])  
                    ]),  
                    html.Div(id="text-analysis-results")  
                ]),  
                dcc.Tab(label='Yelp Reviews Analysis', children=[  
                    dbc.Card([  
                        dbc.CardHeader("Enter Yelp Business"),  
                        dbc.CardBody([  
                            dcc.Input(id="yelp-business-input", placeholder="Enter Yelp Business", type="text", style={'width': '100%'}),  
                            dcc.Dropdown(id="business-suggestions", style={'width': '100%', 'display': 'none'}),  # Hidden initially
                            dbc.Button("Analyze", id="analyze-yelp-btn", color="primary", className="my-2")  
                        ])  
                    ]),  
                    # Wrap the results section with dcc.Loading for the loading indicator
                    dcc.Loading(
                        type="circle",  # You can choose different types for the loading spinner
                        children=html.Div(id="yelp-reviews-results")
                    )
                ])  
            ])  
        ]),  
    ], className="my-4")  
], fluid=True)

# Register callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=4200)
