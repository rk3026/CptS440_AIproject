from dash import Input, Output, State, html, dash_table
import dash
from logic.yelp_data import *
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from logic.yelp_data import *
from collections import Counter
from dash import ctx


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

    @app.callback(
        Output('yelp-reviews-results', 'children'),
        Input({'type': 'business-button', 'index': dash.ALL}, 'n_clicks'),
        State({'type': 'business-button', 'index': dash.ALL}, 'id'),
        prevent_initial_call=True
    )
    def analyze_yelp_reviews(n_clicks_list, ids):
        # Guard: Ignore callback if nothing has actually been clicked
        if not ctx.triggered_id or not any(n > 0 for n in n_clicks_list):
            return dash.no_update

        # Get the business_id from the clicked button
        business_id = ctx.triggered_id['index']

        reviews = load_reviews_for_business_from_db(business_id, limit=100)
        business_info = get_business_info(business_id)

        if not reviews:
            return dbc.Alert("No reviews found for this business.", color="danger")

        sentiment_data = []
        actual_star_counts = Counter()
        predicted_label_counts = Counter()

        for r in reviews:
            stars = int(r.get('stars', 0))

            sentiment_result = analyze_text_sentiment(r['text'])
            if not sentiment_result:
                continue

            predicted_label = sentiment_result[0]['label']
            score = sentiment_result[0]['score']

            if stars in [1, 2, 3, 4, 5]:
                actual_star_counts[f'{stars}_star'] += 1
            if predicted_label in ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]:
                predicted_label_counts[predicted_label] += 1

            sentiment_data.append({
                'Review Text': r['text'],
                'Actual Stars': stars,
                'Predicted Sentiment': predicted_label,
                'Confidence': f"{score:.2f}"
            })

        business_header = html.Div([
            html.H5("Analyzing Reviews for Business:"),
            html.H6(business_info['name']),
            html.P(f"{business_info['address']}, {business_info['city']}, "
                f"{business_info['state']} {business_info['postal_code']}")
        ], className="mb-3")

        summary = html.Div([
            business_header,
            html.H5("Review Summary"),
            html.P(f"Total Reviews Analyzed: {len(sentiment_data)}"),
            html.H6("Actual Star Ratings:"),
            html.Ul([
                html.Li(f"5 Stars: {actual_star_counts['5_star']}"),
                html.Li(f"4 Stars: {actual_star_counts['4_star']}"),
                html.Li(f"3 Stars: {actual_star_counts['3_star']}"),
                html.Li(f"2 Stars: {actual_star_counts['2_star']}"),
                html.Li(f"1 Star: {actual_star_counts['1_star']}")
            ]),
            html.H6("Predicted Sentiment Labels (via BERT):"),
            html.Ul([
                html.Li(f"5 Stars: {predicted_label_counts['5 stars']}"),
                html.Li(f"4 Stars: {predicted_label_counts['4 stars']}"),
                html.Li(f"3 Stars: {predicted_label_counts['3 stars']}"),
                html.Li(f"2 Stars: {predicted_label_counts['2 stars']}"),
                html.Li(f"1 Star: {predicted_label_counts['1 star']}")
            ]),
        ])

        table = dash_table.DataTable(
            columns=[
                {'name': 'Review Text', 'id': 'Review Text'},
                {'name': 'Actual Stars', 'id': 'Actual Stars'},
                {'name': 'Predicted Sentiment', 'id': 'Predicted Sentiment'},
                {'name': 'Confidence', 'id': 'Confidence'}
            ],
            data=sentiment_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'}
        )

        return html.Div([summary, html.Hr(), table])

