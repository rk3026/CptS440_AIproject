from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px

from callbacks.yelp_callbacks import register_yelp_callbacks
from callbacks.text_callbacks import register_text_sentiment_callbacks
from callbacks.social_media_callbacks import register_bluesky_callbacks

# Create an initial empty pie chart with the correct colors
initial_pie = px.pie(
    names=["Positive", "Neutral", "Negative"],
    values=[0, 0, 0],
    title="Overall Sentiment Distribution"
)
initial_pie.update_traces(marker=dict(colors=["green", "grey", "red"]))

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Sentiment Analysis Dashboard"
)
server = app.server

app.layout = dbc.Container([
    # Stores for batched Bluesky comments & sentiments
    dcc.Store(id="bluesky-comments-store"),
    dcc.Store(id="bluesky-sentiments-store"),

    # Interval for polling the next batch
    dcc.Interval(
        id="bluesky-interval",
        interval=500,   # 1000ms = 1s
        disabled=True,   # start disabled
        n_intervals=0
    ),

    # Header
    dbc.Row([
        dbc.Col(
            html.H2("Sentiment Analysis Dashboard", className="text-center my-3"),
            width=True
        )
    ]),

    # Tabs
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id="tabs", children=[

                # Text Analysis Tab
                dcc.Tab(label="Text Analysis", children=[
                    dcc.Loading(
                        id="loading-text",
                        type="circle",
                        children=[
                            dbc.Card([
                                dbc.CardHeader("Enter Text for Analysis"),
                                dbc.CardBody([
                                    dcc.Textarea(
                                        id="input-text",
                                        value="Enter text here",
                                        style={"width": "100%", "height": 100}
                                    ),
                                    dbc.Button(
                                        "Analyze",
                                        id="analyze-btn",
                                        color="primary",
                                        className="mt-2"
                                    )
                                ])
                            ], className="mb-4"),
                            html.Div(
                                id="text-analysis-results",
                                style={"minHeight": "200px"}
                            )
                        ]
                    )
                ]),

                # Yelp Reviews Analysis Tab
                dcc.Tab(label="Yelp Reviews Analysis", children=[
                    dcc.Loading(
                        id="loading-yelp",
                        type="circle",
                        children=[
                            dbc.Card([
                                dbc.CardHeader("Search for a Yelp Business"),
                                dbc.CardBody([
                                    dcc.Input(
                                        id="yelp-business-input",
                                        placeholder="Enter Yelp Business Name",
                                        type="text",
                                        style={"width": "100%"}
                                    ),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="country-dropdown",
                                                options=[
                                                    {"label": c, "value": c}
                                                    for c in ["USA", "Canada", "UK", "Australia"]
                                                ],
                                                placeholder="Select Country",
                                                style={"width": "100%"}
                                            ),
                                            width=12
                                        )
                                    ], className="mt-2"),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="state-dropdown",
                                                options=[],
                                                placeholder="Select State",
                                                style={"width": "100%"}
                                            ),
                                            width=12
                                        )
                                    ], className="mt-2"),
                                    dcc.Dropdown(
                                        id="business-suggestions",
                                        style={"width": "100%", "display": "none"}
                                    ),
                                    dbc.Button(
                                        "Analyze",
                                        id="analyze-yelp-btn",
                                        color="primary",
                                        className="mt-2"
                                    )
                                ])
                            ], className="mb-4"),
                            html.Div(
                                id="yelp-reviews-results",
                                style={"minHeight": "200px"}
                            )
                        ]
                    )
                ]),

                # Bluesky Post Analysis Tab
                dcc.Tab(label="Bluesky Post Analysis", children=[
                    # Input Card outside Loading
                    dbc.Card([
                        dbc.CardHeader("Analyze Comments of Bluesky Post"),
                        dbc.CardBody([
                            dcc.Input(
                                id="bluesky-post-url",
                                placeholder="Enter Bluesky post URL",
                                type="text",
                                style={"width": "100%"}
                            ),
                            dbc.Button(
                                "Analyze Comments",
                                id="analyze-comments-btn",
                                color="primary",
                                className="mt-2"
                            )
                        ])
                    ], className="mb-4"),

                    # Only chart + results in Loading
                    # dcc.Loading(
                    #     id="loading-bluesky",
                    #     type="circle",
                    #     children=[
                           
                    #     ]
                    # )
                    html.Div(id="bluesky-comment-count", className="mb-2 fw-bold text-end"),
                    dcc.Graph(
                            id="sentiment-summary-graph",
                            figure=initial_pie,
                            style={"minHeight": "300px"}
                        ),
                    html.Div(
                        id="bluesky-comment-results",
                        style={"minHeight": "300px"}
                    )
                ])

            ])
        ])
    ], className="my-4"),

], fluid=True)

# Register all callback modules
register_yelp_callbacks(app)
register_text_sentiment_callbacks(app)
register_bluesky_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=4200)
