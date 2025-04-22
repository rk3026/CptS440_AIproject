from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

# Create an initial empty pie chart with the correct colors
initial_pie = px.pie(
    names=["Positive", "Neutral", "Negative"],
    values=[0, 0, 0],
    title="Overall Sentiment Distribution"
)
initial_pie.update_traces(marker=dict(colors=["green", "grey", "red"]))

def get_bluesky_tab():
    return dcc.Tab(label="Bluesky Post Analysis", children=[
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
