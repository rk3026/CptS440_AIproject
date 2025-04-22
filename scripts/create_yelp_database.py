'''
This script creates and fills in a database with business and review info.
It is slow because it only puts information if it has both the business and reviews for.
'''

import sqlite3
import json
import time

def create_db():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('data/yelp_reviews.db')
    cursor = conn.cursor()

    # Create tables for businesses and reviews
    cursor.execute('''CREATE TABLE IF NOT EXISTS businesses (
        business_id TEXT PRIMARY KEY,
        name TEXT,
        address TEXT,
        city TEXT,
        state TEXT,
        postal_code TEXT,
        latitude REAL,
        longitude REAL,
        stars REAL,
        review_count INTEGER,
        is_open INTEGER,
        categories TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        business_id TEXT,
        user_id TEXT,
        stars INTEGER,
        text TEXT,
        FOREIGN KEY(business_id) REFERENCES businesses(business_id)
    )''')

    conn.commit()
    conn.close()

def load_reviews_for_business(business_id, cursor):
    # Open the reviews file with the correct encoding
    with open('data/yelp_dataset/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as file:
        try:
            for line in file:
                review = json.loads(line)
                if review['business_id'] == business_id:
                    # Insert review into the database for this business
                    cursor.execute('''INSERT OR IGNORE INTO reviews (review_id, business_id, user_id, stars, text)
                                    VALUES (?, ?, ?, ?, ?)''', 
                                    (review['review_id'], review['business_id'], review['user_id'], review['stars'], review['text']))
        except sqlite3.Error as e:
            print(f"Error inserting review for business {business_id}: {e}")
            return False
    return True

def load_businesses_and_reviews_to_db():
    conn = sqlite3.connect('data/yelp_reviews.db')
    cursor = conn.cursor()

    # Open the businesses file with the correct encoding
    with open('data/yelp_dataset/yelp_academic_dataset_business.json', 'r', encoding='utf-8') as file:
        total_businesses = 0  # To count how many businesses are processed
        successful_inserts = 0  # To track successful inserts
        failed_inserts = 0  # To track failed inserts

        try:
            for i, line in enumerate(file, start=1):
                business = json.loads(line)

                # Prepare business data for insertion
                business_data = (
                    business['business_id'],
                    business['name'],
                    business.get('address', ''),
                    business['city'],
                    business['state'],
                    business.get('postal_code', ''),
                    business.get('latitude', None),
                    business.get('longitude', None),
                    business['stars'],
                    business['review_count'],
                    business['is_open'],
                    business.get('categories', '')
                )

                try:
                    # Insert business into the database
                    cursor.execute('''INSERT OR IGNORE INTO businesses (
                        business_id, name, address, city, state, postal_code, 
                        latitude, longitude, stars, review_count, is_open, categories
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', business_data)
                    successful_inserts += 1
                    print(f"Business {business['business_id']} inserted...")

                    # Load reviews for this business
                    if load_reviews_for_business(business['business_id'], cursor):
                        print(f"Reviews for business {business['business_id']} inserted...")
                    else:
                        print(f"Failed to insert reviews for business {business['business_id']}")

                    # Print a progress message every 100 businesses
                    if i % 100 == 0:
                        print(f"Processed {i} businesses...")

                except sqlite3.Error as e:
                    print(f"Error inserting business {business['business_id']}: {e}")
                    failed_inserts += 1

                # Sleep a little bit to prevent overwhelming the console (optional)
                time.sleep(0.01)

        except KeyboardInterrupt:
            # Gracefully handle keyboard interrupt (Ctrl+C)
            print("\nProcess interrupted by the user.")
            print(f"\nSummary before stopping:")
            print(f"Total businesses processed: {i}")
            print(f"Successfully inserted: {successful_inserts}")
            print(f"Failed inserts: {failed_inserts}")
            conn.commit()
            conn.close()
            return  # Exit the function if interrupted

        # If all businesses processed, commit the changes
        conn.commit()
        conn.close()

        # Print summary of insert operation
        print(f"\nSummary:")
        print(f"Total businesses processed: {i}")
        print(f"Successfully inserted: {successful_inserts}")
        print(f"Failed inserts: {failed_inserts}")

def main():
    create_db()
    load_businesses_and_reviews_to_db()  # Load businesses and reviews in sequence

if __name__ == '__main__':
    main()
