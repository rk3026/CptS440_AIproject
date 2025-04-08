from yelpapi import YelpAPI
import os
from logic.models import models

# Initialize Yelp API
API_KEY = os.getenv("YELP_API_KEY")  # Make sure your Yelp API Key is set
yelp_api = YelpAPI(API_KEY)

def search_yelp_business(query, state=None, country=None, max_results=5):
    # Prepare the location
    location = query  # Start with the query itself
    
    if state and country:
        location = f"{state}, {country}"  # Combine state and country if both are provided
    elif state:
        location = f"{state}"  # If only state is provided, use it
    elif country:
        location = f"{country}"  # If only country is provided, use it

    # Define search parameters
    params = {
        'term': query,
        'location': location,  # Use the location string
        'limit': max_results
    }

    try:
        # Query Yelp API
        response = yelp_api.search_query(**params)
        businesses = []

        # Process businesses from the response
        for business in response.get('businesses', []):
            business_info = {
                'business_id': business['id'],
                'name': business['name'],
                'address': ' '.join(business['location'].get('address', [])),
                'city': business['location'].get('city', ''),
                'state': business['location'].get('state', ''),
                'postal_code': business['location'].get('zip_code', '')
            }
            businesses.append(business_info)

        return businesses
    except Exception as e:
        print(f"Error fetching Yelp data: {e}")
        return []


# Function to load reviews for a specific business
def load_reviews_for_business(business_id, limit=10):
    reviews_response = yelp_api.reviews_query(business_id)
    
    reviews = []
    for review in reviews_response.get('reviews', [])[:limit]:
        reviews.append({
            'text': review['text'],
            'rating': review['rating'],
            'time_created': review['time_created']
        })
    
    return reviews

# Function to analyze sentiment of text (same as previous)
def analyze_text_sentiment(text):
    max_length = 512
    if len(text.split()) > max_length:
        text = ' '.join(text.split()[:max_length])  # Truncate text to 512 tokens

    model = models["Yelp BERT"]
    sentiment = model(text)
    return sentiment