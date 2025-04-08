from dash import Input, Output, State
from sentiment_analysis import search_yelp_business, load_reviews_for_business, analyze_text_sentiment, models, roberta_label_map
from dash import html

def register_callbacks(app):
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
        
        # Show dropdown if there are suggestions
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
        # Search for businesses matching the entered name (limit to 5 results)
        business_info = search_yelp_business(business_name, max_results=5)

        if business_info:
            business_id = business_info[0]['business_id']  # Take the first match

            # Limit the number of reviews for the business (limit to 10 reviews)
            reviews_for_business = load_reviews_for_business(business_id, limit=10)

            if not reviews_for_business.empty:
                sentiment_results = []
                for _, review in reviews_for_business.iterrows():
                    review_text = review['text']
                    sentiment = analyze_text_sentiment(review_text)  # Sentiment analysis
                    sentiment_results.append(html.Div([html.H5(f"Review: {review_text[:50]}..."),
                                                       html.P(f"Sentiment: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")]))

                return html.Div(sentiment_results)
            else:
                return "No reviews found for this business."
        else:
            return "Business not found."

    # Callback for Text Analysis
    @app.callback(
        Output('text-analysis-results', 'children'),
        Input('analyze-btn', 'n_clicks'),
        State('input-text', 'value'),
        prevent_initial_call=True
    )
    def analyze_text(n_clicks, input_text):
        results = []
        for model_name, model in models.items():
            if isinstance(model, str):  # Skip unsupported models
                results.append(html.Div([html.H5(f'{model_name}: Unsupported')]))
                continue
            
            try:
                prediction = model(input_text)
                label = prediction[0]['label']
                score = round(prediction[0]['score'], 4)

                # Use predefined label map for Twitter RoBERTa
                if model_name == "Twitter RoBERTa":
                    label = roberta_label_map.get(label, label)

                results.append(html.Div([html.H5(f'{model_name}: {label}'), html.P(f'Score: {score}')]))
            except Exception as e:
                results.append(html.Div([html.H5(f'{model_name}: Error - {str(e)}')]))

        return html.Div(results)
