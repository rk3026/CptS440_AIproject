from dash import dcc, html
import dash_bootstrap_components as dbc
from layout.tabs.text_tab import get_text_tab
from layout.tabs.yelp_tab import get_yelp_tab
from layout.tabs.bluesky_tab import get_bluesky_tab

def get_main_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Sentiment Analysis Dashboard", className="text-center my-auto"), width=True),
        ], align="center", className="my-3"),

        dbc.Row([
            dbc.Col([
                dcc.Tabs(id="tabs", children=[
                    get_text_tab(),
                    get_yelp_tab(),
                    get_bluesky_tab()
                ])
            ])
        ], className="my-4")
    ], fluid=True)
