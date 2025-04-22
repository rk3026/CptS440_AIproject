from dash import Input, Output, State, html, dash_table
from logic.yelp_data import *
import dash_bootstrap_components as dbc

def register_yelp_callbacks(app):
    # Update suggestions dropdown based on user input
    @app.callback(
        Output('business-suggestions', 'options'),
        Output('business-suggestions', 'style'),
        Input('yelp-business-input', 'value'),
        Input('state-dropdown', 'value'),
        Input('city-dropdown', 'value'),
        prevent_initial_call=True
    )
    # Update state dropdown based on selected country
    @app.callback(
        Output('state-dropdown', 'options'),
        Input('country-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_state_dropdown(country):
        if not country:
            return []
        states = get_available_states(country)
        return [{'label': s, 'value': s} for s in states]

    # Update city dropdown based on selected state
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

    # Analyze reviews and return sentiment table + summary
    @app.callback(
        Output('yelp-reviews-results', 'children'),
        Input('analyze-yelp-btn', 'n_clicks'),
        State('business-suggestions', 'value'),
        prevent_initial_call=True
    )
    def analyze_yelp_reviews(n_clicks, business_id):
        if not business_id:
            return "Please select a business."

        reviews = load_reviews_for_business_from_db(business_id, limit=100)  # adjust limit if needed

        if not reviews:
            return "No reviews found for this business."

        # Analyze sentiments
        sentiment_data = []
        score_summary = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'count': 0}

        for r in reviews:
            sentiment = analyze_text_sentiment(r['text'])[0]  # Assume 1 output per review
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

        # Summary card
        summary = html.Div([
            html.H5("Sentiment Summary"),
            html.P(f"Total Reviews: {score_summary['count']}"),
            html.P(f"Positive: {score_summary['POSITIVE']}"),
            html.P(f"Neutral: {score_summary['NEUTRAL']}"),
            html.P(f"Negative: {score_summary['NEGATIVE']}")
        ])

        # Reviews table
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
