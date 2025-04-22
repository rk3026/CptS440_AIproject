import sqlite3
from logic.models import models

# === Constants ===
YELP_REVIEW_DB = "data/yelp_reviews.db"

TABLE_BUSINESSES = "businesses"
TABLE_REVIEWS = "reviews"

# === DB Access Functions ===

def get_all_states():
    conn = sqlite3.connect(YELP_REVIEW_DB)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT state FROM {TABLE_BUSINESSES} ORDER BY state")
    states = [row[0] for row in cursor.fetchall()]
    conn.close()
    return states

def get_available_cities(state):
    conn = sqlite3.connect(YELP_REVIEW_DB)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT city FROM {TABLE_BUSINESSES} WHERE state = ? ORDER BY city", (state,))
    cities = [row[0] for row in cursor.fetchall()]
    conn.close()
    return cities

def search_yelp_business_from_db(business_name, state=None, city=None, max_results=10):
    conn = sqlite3.connect(YELP_REVIEW_DB)
    cursor = conn.cursor()

    query = f'''
        SELECT business_id, name, address, city, state
        FROM {TABLE_BUSINESSES}
        WHERE name LIKE ?
    '''
    params = [f"%{business_name}%"]

    if state:
        query += " AND state = ?"
        params.append(state)

    if city:
        query += " AND city = ?"
        params.append(city)

    query += " LIMIT ?"
    params.append(max_results)

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    businesses = []
    for row in results:
        businesses.append({
            'business_id': row[0],
            'name': row[1],
            'location': {
                'address1': row[2],
                'city': row[3],
                'state': row[4]
            }
        })
    return businesses

def load_reviews_for_business_from_db(business_id, limit=None):
    conn = sqlite3.connect(YELP_REVIEW_DB)
    cursor = conn.cursor()

    query = f'SELECT text FROM {TABLE_REVIEWS} WHERE business_id = ?'
    params = [business_id]

    if limit:
        query += ' LIMIT ?'
        params.append(limit)

    cursor.execute(query, params)
    reviews = [{'text': row[0]} for row in cursor.fetchall()]
    conn.close()
    return reviews

# === Sentiment Analysis ===

def analyze_text_sentiment(text):
    max_length = 512
    if len(text.split()) > max_length:
        text = ' '.join(text.split()[:max_length])  # Truncate text to 512 tokens

    model = models["Yelp BERT"]
    sentiment = model(text)
    return sentiment
