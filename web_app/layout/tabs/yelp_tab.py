from dash import dcc, html
import dash_bootstrap_components as dbc

def get_yelp_tab():
    return dcc.Tab(label='Yelp Reviews Analysis', className="yelp-tab", children=[
        # Logo
        html.Div([
            html.Img(src="assets/Yelp_Logo.svg", style={"height": "60px"}),
        ], className="mb-3 text-center"),
        
        dbc.Card([
            dbc.CardHeader("Find and Analyze Yelp Business Reviews"),
            dbc.CardBody([

                html.Div("Step 1: Choose a location", className="mb-2 fw-bold"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id="country-dropdown",
                        options=[{'label': 'United States', 'value': 'US'}],
                        value='US',
                        disabled=True,
                        style={'backgroundColor': '#f8f9fa'}
                    )),
                    dbc.Col(dcc.Dropdown(id="state-dropdown", options=[], placeholder="Select State"), width=4),
                    dbc.Col(dcc.Dropdown(id="city-dropdown", options=[], placeholder="Select City"), width=4),
                ], className="mb-3"),

                html.Div("Step 2: Search for a business name", className="mb-2 fw-bold"),
                dcc.Input(id="yelp-business-input", type="text", placeholder="Enter part of business name...", style={'width': '100%'}),

                html.Div("Matching Businesses:", className="mt-4 mb-1 fw-bold"),
                html.Div(id="business-list"),

            ])
        ]),
        dcc.Loading(
            type="circle",
            children=html.Div(id="yelp-reviews-results", className="mt-4")
        )
    ])
