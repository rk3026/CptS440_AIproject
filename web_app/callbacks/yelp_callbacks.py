from dash import Input, Output, State, html, dcc, ctx
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from logic.yelp_data import *
from logic.models import label_colors
from collections import Counter

def summarize_review(text):
    return text[:80] + ("..." if len(text) > 80 else "")

def register_yelp_callbacks(app):
    @app.callback(
        Output('yelp-reviews-results', 'children', allow_duplicate=True),
        Input({'type': 'business-button', 'index': dash.ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_reviews(_):
        return ""

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

        return dbc.ListGroup([
                    dbc.ListGroupItem(
                        html.Div([
                            html.H6(b['name'], className="mb-1 fw-bold"),
                            html.Small(f"{b['location']['city']}, {b['location']['state']}", className="text-muted")
                        ]),
                        id={'type': 'business-button', 'index': b['business_id']},
                        action=True,
                        n_clicks=0,
                        style={"cursor": "pointer"}
                    )
                    for b in matches
                ], flush=True)


    @app.callback(
        Output('yelp-reviews-results', 'children'),
        Input({'type': 'business-button', 'index': dash.ALL}, 'n_clicks'),
        State({'type': 'business-button', 'index': dash.ALL}, 'id'),
        prevent_initial_call=True
    )
    def analyze_yelp_reviews(n_clicks_list, ids):
        if not ctx.triggered_id or not any(n > 0 for n in n_clicks_list):
            return dash.no_update

        business_id = ctx.triggered_id['index']
        reviews = load_reviews_for_business_from_db(business_id, limit=100)
        business_info = get_business_info(business_id)

        if not reviews:
            return dbc.Alert("No reviews found for this business.", color="danger")

        sentiment_data = []
        actual_star_counts = Counter()
        predicted_label_counts = Counter()

        for i, r in enumerate(reviews):
            stars = int(r.get('stars', 0))
            sentiment_result = analyze_text_sentiment(r['text'])
            if not sentiment_result:
                continue

            predicted_label = sentiment_result[0]['label']
            score = sentiment_result[0]['score']

            actual_star_counts[f"{stars} Stars"] += 1
            predicted_label_counts[predicted_label] += 1

            sentiment_data.append({
                "id": i,
                "text": r["text"],
                "summary": summarize_review(r["text"]),
                "stars": stars,
                "predicted": predicted_label,
                "score": score
            })

        pie_actual = px.pie(
            names=list(actual_star_counts.keys()),
            values=list(actual_star_counts.values()),
            title="Actual Yelp Star Ratings",
            color=list(actual_star_counts.keys()),
            color_discrete_map={k: label_colors.get(k.lower(), "lightblue") for k in actual_star_counts}
        )

        pie_predicted = px.pie(
            names=list(predicted_label_counts.keys()),
            values=list(predicted_label_counts.values()),
            title="Predicted Sentiment Labels",
            color=list(predicted_label_counts.keys()),
            color_discrete_map={k: label_colors.get(k.lower(), "lightblue") for k in predicted_label_counts}
        )

        for fig in [pie_actual, pie_predicted]:
            fig.update_layout(
                height=300,
                margin=dict(t=40, l=0, r=0, b=20),
                paper_bgcolor="#FDFCFB",
                plot_bgcolor="#FDFCFB",
                font=dict(color="#3A3129", family="Segoe UI")
            )

        pie_row = dbc.Row([
            dbc.Col(dcc.Graph(figure=pie_actual), md=6),
            dbc.Col(dcc.Graph(figure=pie_predicted), md=6)
        ], className="mb-4")

        grouped_by_star = {i: [] for i in range(5, 0, -1)}
        modals = []

        for item in sentiment_data:
            i = item["id"]
            rid = {'type': 'review-modal', 'index': i}

            star_key = f"{item['stars']} stars"
            star_color = label_colors.get(star_key.lower(), "black")
            pred_color = label_colors.get(item['predicted'].lower(), "black")

            card = html.Div(
                dbc.Card(
                    dbc.CardBody([
                        html.H6([
                            html.Span(f"{item['stars']} Stars", style={"color": star_color}),
                            " | ",
                            html.Span(f"Predicted: {item['predicted']} ({item['score']:.2f})", style={"color": pred_color})
                        ], className="fw-bold mb-2"),
                        html.P(item["summary"], className="card-text")
                    ]),
                    className="review-card",
                    style={
                        "borderColor": "#826B55",
                        "backgroundColor": "#F8F4F2",
                        "borderRadius": "10px"
                    }
                ),
                id={'type': 'review-wrap', 'index': i},
                n_clicks=0,
                style={"cursor": "pointer"}
            )

            grouped_by_star[item['stars']].append(card)

            modals.append(
                dbc.Modal([
                    dbc.ModalHeader(html.H4("Review Details", className="fw-bold text-primary")),
                    dbc.ModalBody([
                        html.P([
                            html.Span("Actual Stars: ", className="fw-bold text-dark"),
                            html.Span(f"{item['stars']}", style={"color": star_color, "fontWeight": "bold"})
                        ]),
                        html.P([
                            html.Span("Predicted: ", className="fw-bold text-dark"),
                            html.Span(item['predicted'], style={"color": pred_color, "fontWeight": "bold"})
                        ]),
                        html.P([
                            html.Span("Confidence: ", className="fw-bold text-dark"),
                            f"{item['score']:.2f}"
                        ]),
                        html.P([
                            html.Span("Summary: ", className="fw-bold text-dark"),
                            "Coming soon..."
                        ]),
                        html.Hr(),
                        html.P([
                            html.Span("Review Text: ", className="fw-bold text-dark"),
                            item["text"]
                        ])
                    ])
                ],
                id=rid,
                is_open=False,
                style={"backgroundColor": "rgba(0, 0, 0, 0.35)"},
                backdrop=True  # enables the modal overlay
                )
            )

        return html.Div([
            html.H2(f"Analyzing Reviews for: {business_info['name']}"),
            html.H6(f"{business_info['address']}, {business_info['city']}, {business_info['state']} {business_info['postal_code']}"),
            html.Hr(),
            html.H4(f"Total Reviews Analyzed: {len(sentiment_data)}"),
            pie_row,
            *[
                html.Div([
                    html.H5(f"{stars} Star Reviews", className="mt-4 mb-2 text-dark fw-bold"),
                    html.Div(grouped_by_star[stars], className="card-grid")
                ]) for stars in grouped_by_star if grouped_by_star[stars]
            ],
            *modals
        ])

    @app.callback(
        Output({'type': 'review-modal', 'index': dash.MATCH}, 'is_open'),
        Input({'type': 'review-wrap', 'index': dash.MATCH}, 'n_clicks'),
        Input({'type': 'review-modal', 'index': dash.MATCH}, 'n_dismiss'),
        State({'type': 'review-modal', 'index': dash.MATCH}, 'is_open'),
        prevent_initial_call=True
    )
    def toggle_modal(open_clicks, dismiss_clicks, is_open):
        return not is_open
