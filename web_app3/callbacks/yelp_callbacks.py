from dash import Input, Output, State, html, dash_table
from logic.yelp_data import search_yelp_business, load_reviews_for_business, analyze_text_sentiment
import dash_bootstrap_components as dbc

def register_yelp_callbacks(app):
    # Callback to display business suggestions as user types
    @app.callback(
        Output('business-suggestions', 'options'),
        Output('business-suggestions', 'style'),
        Input('yelp-business-input', 'value'),
        Input('state-dropdown', 'value'),
        Input('country-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_suggestions(input_value, state, country):
        if not input_value:
            return [], {'display': 'none'}
        
        businesses = search_yelp_business(input_value, state=state, country=country, max_results=5)
        suggestions = []

        for business in businesses:
            business_name = business.get('name', 'Unknown Business')
            
            location = "Location not available"
            if 'location' in business:
                address = business['location'].get('address1', '')
                city = business['location'].get('city', '')
                state = business['location'].get('state', '')
                if address or city or state:
                    location = f"{address}, {city}, {state}"

            suggestions.append({
                'label': f"{business_name} - {location}",
                'value': business_name
            })
        
        return suggestions, {'display': 'block'}

    # Callback to populate the input field when a business name suggestion is selected
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
        # Search for the business by name
        business_info = search_yelp_business(business_name, max_results=5)

        if business_info:
            business = business_info[0]
            business_id = business['business_id']
            print(f"Checking reviews for business_id: {business_id}")  # Debugging the business_id
            
            try:
                # Attempt to load reviews for the business
                reviews_for_business = load_reviews_for_business(business_id, limit=10)

                if reviews_for_business:
                    # Prepare the data for the table
                    review_data = []
                    for review in reviews_for_business:
                        review_text = review['text']
                        sentiment = analyze_text_sentiment(review_text)

                        # Add a row for each review with sentiment
                        review_data.append({
                            'Review Text': review_text,
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

            except Exception as e:
                # Handle the error gracefully if reviews are not found or API call fails
                return f"Error loading reviews: {str(e)}"
        else:
            return "Business not found."
