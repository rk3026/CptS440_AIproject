from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from callbacks.yelp_callbacks import register_yelp_callbacks
from callbacks.text_callbacks import register_text_sentiment_callbacks
from callbacks.social_media_callbacks import register_bluesky_callbacks

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
                            dcc.Textarea(id="input-text", value="Enter text here", style={'width': '100%', 'height': 100}),  
                            dbc.Button("Analyze", id="analyze-btn", color="primary", className="my-2")  
                        ])  
                    ]),  

                    # Wrap the results of the text sentiment analysis in dcc.Loading
                    dcc.Loading(
                        type="circle",  # You can also use "dot" or "default" here
                        children=html.Div(id="text-analysis-results")
                    )
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
                    dcc.Loading(
                        type="circle",
                        children=html.Div(id="yelp-reviews-results")
                    )
                ]),

                # Add a tab for Bluesky post and comments analysis
                dcc.Tab(label='Bluesky Post Analysis', children=[  
                    # Section to analyze comments of an existing Bluesky post
                    dbc.Card([  
                        dbc.CardHeader("Analyze Comments of Bluesky Post"),  
                        dbc.CardBody([  
                            dcc.Input(id="bluesky-post-url", placeholder="Enter Bluesky post URL", type="text", style={'width': '100%'}),  
                            dbc.Button("Analyze Comments", id="analyze-comments-btn", color="primary", className="my-2")  
                        ])  
                    ]),

                    # Wrap the result section with `dcc.Loading` to show a loading spinner
                    dcc.Loading(
                        type="circle",  # You can change the type to "dot" or "default" if you prefer
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
register_bluesky_callbacks(app)  # Register Bluesky callbacks

if __name__ == "__main__":
    app.run(debug=True, port=4200)
