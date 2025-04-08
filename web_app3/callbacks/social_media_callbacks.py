from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from atproto import Client
import os
from dotenv import load_dotenv
from logic.models import models, roberta_label_map

# Load environment variables
load_dotenv()

# Initialize the Bluesky Client
client = Client()
client.login(os.getenv('BLUESKY_USER'), os.getenv('BLUESKY_APP_KEY'))

# Function to extract profile ID and post ID from the Bluesky URL
# Example url: https://bsky.app/profile/rk3026.bsky.social/post/3lmdal5hrho2n
def extract_post_id(post_url):
    # Split the URL by "/"
    parts = post_url.split('/')
    user_did = parts[4]
    profile_id = get_profile_id_from_user_did(user_did)
    post_id = parts[-1]
    # Construct AT-URI for the post
    return f"at://{profile_id}/app.bsky.feed.post/{post_id}"

def get_profile_id_from_user_did(user_did):
    profile = client.get_profile(user_did)  # This fetches the profile for the given DID
    profile_id = profile['did']  # The profile ID is part of the 'did' field in the response
    return profile_id

# Register Bluesky login, post, and comment analysis functionality
def register_bluesky_callbacks(app):
    # Callback to analyze comments of a Bluesky post based on the post link provided
    @app.callback(
        Output('bluesky-comment-results', 'children'),
        Input('analyze-comments-btn', 'n_clicks'),
        State('bluesky-post-url', 'value'),
        prevent_initial_call=True
    )
    def analyze_comments_of_post(n_clicks, post_url):
        try:
            # Extract post ID from URL
            post_id = extract_post_id(post_url)

            # Fetch the post's comment thread
            post_thread = client.get_post_thread(post_id).thread
            
            comments = post_thread.replies

            if not comments:
                return html.Div([html.H5("No comments found for this post.")])

            # Analyze sentiment of each comment and its replies
            sentiment_results = []

            # Function to recursively analyze replies
            def analyze_replies(post, parent_text=None):
                replies = post.get('replies', [])
                for reply in replies:
                    # Check the structure of the reply object to ensure we access the text correctly
                    reply_text = reply['post']['record']['text'] if 'post' in reply else 'No text'
                    sentiment = analyze_text_sentiment(reply_text)
                    sentiment_results.append(html.Div([
                        html.H5(f"Reply to {parent_text[:100]}..."),
                        html.P(f"Reply: {reply_text[:100]}..."),
                        html.P(f"Sentiment: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")
                    ]))

            # Iterate over the top-level comments
            for comment in comments:
                comment_text = comment.post.record.text
                sentiment = analyze_text_sentiment(comment_text)
                sentiment_results.append(html.Div([
                    html.H5(f"Comment: {comment_text[:100]}..."),
                    html.P(f"Sentiment: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")
                ]))

                # Analyze replies to this comment
                #analyze_replies(comment['post'], parent_text=comment_text)                    
            return html.Div(sentiment_results)

        except Exception as e:
            return html.Div([html.H5("Error analyzing comments"),
                            html.P(str(e))])

# Function to analyze sentiment of text
def analyze_text_sentiment(text):
    # Limit the text to 512 tokens (since BERT models like Yelp BERT have a max length of 512 tokens)
    max_length = 512
    if len(text.split()) > max_length:
        text = ' '.join(text.split()[:max_length])  # Truncate text to 512 tokens

    # Get sentiment using the model
    model = models["Twitter RoBERTa"]
    sentiment = model(text)
    
    # Get the label from the first result
    label = sentiment[0]['label']
    
    # Map the label to the predefined human-readable labels using the mapping
    mapped_label = roberta_label_map.get(label, 'Unknown')  # Default to 'Unknown' if not found in the map

    # Sentiment score
    score = sentiment[0]['score']
    
    return [{'label': mapped_label, 'score': score}]

