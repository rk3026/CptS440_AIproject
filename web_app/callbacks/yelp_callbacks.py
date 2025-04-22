from dash import Input, Output, State, html, dash_table
import dash
from logic.yelp_data import *
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from logic.yelp_data import *

def register_yelp_callbacks(app):
    # Clear results on any button click
    @app.callback(
        Output('yelp-reviews-results', 'children', allow_duplicate=True),
        Input({'type': 'business-button', 'index': dash.ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_reviews(_):
        return ""  # Clears the content (and triggers spinner next)

    @app.callback(
        Output('state-dropdown', 'options'),
        Input('country-dropdown', 'value'),
    )
    def update_state_dropdown(country):
        if not country:
            return []
        states = get_all_states()
        return [{'label': s, 'value': s} for s in states]

    @app.callback(
        Output('city-dropdown', 'options'),
        Input('state-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_city_dropdown(state):
        if not state:
            return []
        cities = get_available_cities(state)
        return [{'label': c, 'value': c} for c in cities]

    @app.callback(
        Output('business-list', 'children'),
        Input('yelp-business-input', 'value'),
        Input('state-dropdown', 'value'),
        Input('city-dropdown', 'value'),
    )
    def update_business_list(name_input, state, city):
        name_input = name_input or ""

        matches = search_yelp_business_from_db(name_input, state, city, max_results=20)

        if not matches:
            return html.Div("No businesses found.", className="text-danger")

        return html.Ul([
            html.Li(
                html.Button(
                    f"{b['name']} ({b['location']['city']}, {b['location']['state']})",
                    id={'type': 'business-button', 'index': b['business_id']},
                    n_clicks=0,
                    className="btn btn-link"
                )
            )
            for b in matches
        ])


    # Callback to analyze reviews when a business button is clicked
    @app.callback(
        Output('yelp-reviews-results', 'children'),
        Input({'type': 'business-button', 'index': dash.ALL}, 'n_clicks'),
        State({'type': 'business-button', 'index': dash.ALL}, 'id'),
        prevent_initial_call=True
    )
    def analyze_yelp_reviews(n_clicks_list, ids):
        # Find the index (business_id) of the button that was clicked
        triggered = [i for i, n in enumerate(n_clicks_list) if n > 0]
        if not triggered:
            return

        idx = triggered[0]
        business_id = ids[idx]['index']

        reviews = load_reviews_for_business_from_db(business_id, limit=100)

        if not reviews:
            return dbc.Alert("No reviews found for this business.", color="danger")

        sentiment_data = []
        score_summary = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'count': 0}

        for r in reviews:
            sentiment = analyze_text_sentiment(r['text'])[0]
            label = sentiment['label'].upper()
            score = sentiment['score']
            sentiment_data.append({
                'Review Text': r['text'],
                'Sentiment': label,
                'Score': f"{score:.2f}"
            })
            score_summary['count'] += 1
            if label in score_summary:
                score_summary[label] += 1

        summary = html.Div([
            html.H5("Sentiment Summary"),
            html.P(f"Total Reviews: {score_summary['count']}"),
            html.P(f"Positive: {score_summary['POSITIVE']}"),
            html.P(f"Neutral: {score_summary['NEUTRAL']}"),
            html.P(f"Negative: {score_summary['NEGATIVE']}")
        ])

        table = dash_table.DataTable(
            columns=[
                {'name': 'Review Text', 'id': 'Review Text'},
                {'name': 'Sentiment', 'id': 'Sentiment'},
                {'name': 'Score', 'id': 'Score'}
            ],
            data=sentiment_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'}
        )

        return html.Div([summary, html.Hr(), table])

