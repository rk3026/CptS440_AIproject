from dash import dcc, html
import dash_bootstrap_components as dbc

def get_bluesky_tab():
    return dcc.Tab(label='Bluesky Post Analysis', children=[
        dbc.Card([
            dbc.CardHeader("Analyze Comments of Bluesky Post"),
            dbc.CardBody([
                dcc.Input(
                    id="bluesky-post-url",
                    placeholder="Enter Bluesky post URL",
                    type="text",
                    style={'width': '100%'}
                ),
                dbc.Button("Analyze Comments", id="analyze-comments-btn", color="primary", className="my-2")
            ])
        ]),

        dcc.Loading(
            type="circle",
            children=html.Div(id="bluesky-comment-results")
        )
    ])
