from dash import Input, Output, State
from logic.yelp_data import search_yelp_business, load_reviews_for_business, analyze_text_sentiment
from dash import html
from dash import dash_table

def register_yelp_callbacks(app):
    # Callback to display business suggestions as user types
    @app.callback(
        Output('business-suggestions', 'options'),
        Output('business-suggestions', 'style'),
        Input('yelp-business-input', 'value'),
        prevent_initial_call=True
    )
    def update_suggestions(input_value):
        if not input_value:
            return [], {'display': 'none'}
        
        businesses = search_yelp_business(input_value, max_results=5)
        suggestions = [{'label': business['name'], 'value': business['name']} for business in businesses]
        
        return suggestions, {'display': 'block'}

    # Callback to populate the input field when a suggestion is selected
    @app.callback(
        Output('yelp-business-input', 'value'),
        Input('business-suggestions', 'value')
    )
    def select_business(selected_business):
        return selected_business

    # Callback for Yelp Reviews Analysis
    @app.callback(
        Output('yelp-reviews-results', 'children'),
        Input('analyze-yelp-btn', 'n_clicks'),
        State('yelp-business-input', 'value'),
        prevent_initial_call=True
    )
    def analyze_yelp_reviews(n_clicks, business_name):
        business_info = search_yelp_business(business_name, max_results=5)

        if business_info:
            business_id = business_info[0]['business_id']  # Take the first match
            reviews_for_business = load_reviews_for_business(business_id, limit=10)

            if not reviews_for_business.empty:
                # Prepare the data for the table
                review_data = []
                for _, review in reviews_for_business.iterrows():
                    review_text = review['text']
                    sentiment = analyze_text_sentiment(review_text)

                    # Add a row for each review with sentiment
                    review_data.append({
                        'Review Text': review_text[:100] + '...',  # Limit preview text
                        'Sentiment': sentiment[0]['label'],
                        'Score': f"{sentiment[0]['score']:.2f}"
                    })

                # Create a table to display reviews and sentiment
                table = dash_table.DataTable(
                    columns=[
                        {'name': 'Review Text', 'id': 'Review Text'},
                        {'name': 'Sentiment', 'id': 'Sentiment'},
                        {'name': 'Score', 'id': 'Score'}
                    ],
                    data=review_data,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                    style_data={'whiteSpace': 'normal', 'height': 'auto'}
                )

                return table
            else:
                return "No reviews found for this business."
        else:
            return "Business not found."

