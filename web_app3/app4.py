from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from callbacks.yelp_callbacks import register_yelp_callbacks
from callbacks.text_callbacks import register_text_sentiment_callbacks
from callbacks.social_media_callbacks import register_bluesky_callbacks

# Define Dash app and layout
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='Sentiment Analysis Dashboard',
    suppress_callback_exceptions=True
)
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
                            dcc.Textarea(id="input-text", value="Enter text here", style={'width': '100%', 'height': 100}),  
                            dbc.Button("Analyze", id="analyze-btn", color="primary", className="my-2")  
                        ])  
                    ]),  

                    dcc.Loading(
                        type="circle",
                        children=html.Div(id="text-analysis-results")
                    )
                ]),  

                dcc.Tab(label='Yelp Reviews Analysis', children=[
                    dbc.Card([
                        dbc.CardHeader("Find and Analyze Yelp Business Reviews"),
                        dbc.CardBody([
                            html.Div("Step 1: Choose a location", className="mb-2 fw-bold"),

                            dbc.Row([
                                dbc.Col(dcc.Dropdown(
                                    id="country-dropdown",
                                    options=[{'label': country, 'value': country} for country in ["USA", "Canada", "UK", "Australia"]],
                                    placeholder="Select Country",
                                ), width=4),

                                dbc.Col(dcc.Dropdown(
                                    id="state-dropdown",
                                    options=[],
                                    placeholder="Select State",
                                ), width=4),

                                dbc.Col(dcc.Dropdown(
                                    id="city-dropdown",
                                    options=[],
                                    placeholder="Select City",
                                ), width=4),
                            ], className="mb-3"),

                            html.Div("Step 2: Search for a business name", className="mb-2 fw-bold"),
                            dcc.Input(id="yelp-business-input", type="text", placeholder="Enter part of business name...", style={'width': '100%'}),
                            
                            html.Div("Suggestions:", className="mt-3 mb-1 fw-bold"),
                            dcc.Dropdown(id="business-suggestions", placeholder="Choose a business", style={'width': '100%'}),

                            dbc.Button("Analyze Reviews", id="analyze-yelp-btn", color="primary", className="mt-3")
                        ])
                    ]),

                    dcc.Loading(
                        type="circle",
                        children=html.Div(id="yelp-reviews-results", className="mt-4")
                    )
                ]),


                dcc.Tab(label='Bluesky Post Analysis', children=[  
                    dbc.Card([  
                        dbc.CardHeader("Analyze Comments of Bluesky Post"),  
                        dbc.CardBody([  
                            dcc.Input(id="bluesky-post-url", placeholder="Enter Bluesky post URL", type="text", style={'width': '100%'}),  
                            dbc.Button("Analyze Comments", id="analyze-comments-btn", color="primary", className="my-2")  
                        ])  
                    ]), 

                    dcc.Loading(
                        type="circle", 
                        children=html.Div(id="bluesky-comment-results")
                    )
                ])
            ])  
        ]),  
    ], className="my-4")  
], fluid=True)

# Register callbacks
register_yelp_callbacks(app)
register_text_sentiment_callbacks(app)
register_bluesky_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=4200)
