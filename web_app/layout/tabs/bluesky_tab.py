from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

# Create an initial empty pie chart with the correct colors
# initial_pie = px.pie(
#     names=["Positive", "Neutral", "Negative"],
#     values=[0, 0, 0],
#     title="Overall Sentiment Distribution"
# )
# initial_pie.update_layout(
#     paper_bgcolor="#FDFCFB",
#     plot_bgcolor="#FDFCFB",
#     font=dict(family="Segoe UI", color="#3A3129"),
#     title=dict(
#         text="Overall Sentiment Distribution",
#         font=dict(size=22, color="#3A3129", family="Segoe UI"),
#         x=0.5,
#         xanchor='center'
#     ),
#     showlegend=False,
#     margin=dict(t=60, l=20, r=20, b=60),
#     uniformtext_minsize=12,
#     uniformtext_mode='hide'
# )
# initial_pie.update_traces(
#     marker=dict(colors=["green", "grey", "red"],
#         line=dict(color='#826b55', width=4)
#     ),
#     textinfo='none',
#     hoverinfo='none',
#     domain=dict(x=[0, 1], y=[0, 1])
# )

def get_bluesky_tab():
    return dcc.Tab(label="Bluesky Post Analysis", className="bluesky-tab", children=[
        # Logo
        html.Div([
            html.Img(src="assets/Bluesky_Logo.svg", style={"height": "60px"}),
        ], className="mb-3 text-center"),
        
        # Hidden Stores
        dcc.Store(id="bluesky-comments-store"),
        dcc.Store(id="bluesky-sentiments-store"),

        # Polling Interval
        dcc.Interval(
            id="bluesky-interval",
            interval=700,
            disabled=True,
            n_intervals=0
        ),

        # Input Card
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

        # Loading for Graph + Results

        html.Div(
            children=[
            html.H5("Total comment processed: ", className="fw-bold"),
            dcc.Loading(
                id="loading-bluesky",
                type="default",
                children=[
                html.Span(id="bluesky-comment-count")
                ],
                className="loading-inline"
            )
            ],
            className="mb-2 fw-bold text-end"
        ),

        #html.Div(id="bluesky-comment-count", className="mb-2 fw-bold text-end"),
        
        dcc.Graph(
            id="sentiment-summary-graph",
            # figure=initial_pie,
            className="graph-container",
            style={"minHeight": "100px"}
        ),
        html.Br(),
        html.H4("Comments: ", className="fw-bold"),
        html.Div(
            id="bluesky-comment-results",
            style={'maxHeight': '75vh', 'overflowY': 'auto', "minHeight": "300px"}
        )
    ])
