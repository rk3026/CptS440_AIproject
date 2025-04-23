from dash import dcc, html
import dash_bootstrap_components as dbc

def get_text_tab():
    return dcc.Tab(label='Text Analysis', className="text-tab", children=[
        dbc.Card([
            dbc.CardHeader("Enter Text for Analysis"),
            dbc.CardBody([
                dcc.Textarea(
                    id="input-text",
                    value="Enter text here",
                    style={'width': '100%', 'height': 100}
                ),
                dbc.Button("Analyze", id="analyze-btn", color="primary", className="my-2")
            ])
        ]),
        
        html.Br(),
        html.H4("Analysis by Models:", className="text-center"),

        dcc.Loading(
            type="circle",
            children=html.Div(id="text-analysis-results")
        )
    ])
