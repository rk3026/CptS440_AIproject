import json
import pandas as pd
from logic.models import models

# Path to the Yelp dataset
YELP_BUSINESS_FILE = './data/yelp_dataset/yelp_academic_dataset_business.json'
YELP_REVIEW_FILE = './data/yelp_dataset/yelp_academic_dataset_review.json'

# Function to search Yelp businesses
def search_yelp_business(query, max_results=5):
    matching_businesses = []
    with open(YELP_BUSINESS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line)
            # Check if the business name matches the query
            if query.lower() in business['name'].lower():
                # Ensure all location fields are included
                business_info = {
                    'business_id': business['business_id'],
                    'name': business['name'],
                    'address': business.get('address', ''),
                    'city': business.get('city', ''),
                    'state': business.get('state', ''),
                    'postal_code': business.get('postal_code', '')
                }
                matching_businesses.append(business_info)

            # Limit the number of results
            if len(matching_businesses) >= max_results:
                break
    return matching_businesses

# Function to load reviews for a specific business
def load_reviews_for_business(business_id, limit=10):
    reviews = []
    with open(YELP_REVIEW_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            if review['business_id'] == business_id:
                reviews.append(review)
                if len(reviews) >= limit:
                    break
    return pd.DataFrame(reviews)

# Function to analyze sentiment of text
def analyze_text_sentiment(text):
    # Limit the text to 512 tokens (since BERT models like Yelp BERT have a max length of 512 tokens)
    max_length = 512
    if len(text.split()) > max_length:
        text = ' '.join(text.split()[:max_length])  # Truncate text to 512 tokens

    model = models["Yelp BERT"]
    sentiment = model(text)
    return sentiment
