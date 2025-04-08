import json
import pandas as pd
from transformers import pipeline

# Define sentiment analysis models
models = {
    "Twitter RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
    "Yelp BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
    "llama3": "x",
    "deepseek": "y",
    "GPT-90": "z"
}

# Predefine the label mapping for Twitter RoBERTa
roberta_label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Path to the Yelp dataset
YELP_BUSINESS_FILE = './data/yelp_dataset/yelp_academic_dataset_business.json'
YELP_REVIEW_FILE = './data/yelp_dataset/yelp_academic_dataset_review.json'

# Function to search Yelp businesses
def search_yelp_business(query, max_results=5):
    matching_businesses = []
    with open(YELP_BUSINESS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line)
            if query.lower() in business['name'].lower():
                matching_businesses.append({
                    'business_id': business['business_id'],
                    'name': business['name']
                })
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
    model = models["Yelp BERT"]
    sentiment = model(text)
    return sentiment
