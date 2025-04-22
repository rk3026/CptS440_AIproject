from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
from atproto import Client
import os
from dotenv import load_dotenv
from logic.models import models, roberta_label_map
from collections import Counter
import plotly.express as px

# Load environment variables
load_dotenv()

# Initialize the Bluesky Client
client = Client()
client.login(os.getenv('BLUESKY_USER'), os.getenv('BLUESKY_APP_KEY'))

# Note: To show a loading spinner on every analyze click, wrap the output placeholder in your layout:
# dcc.Loading(id='loading-comments', type='default', children=html.Div(id='bluesky-comment-results'))

# Function to extract profile ID and post ID from the Bluesky URL
def extract_post_id(post_url):
    parts = post_url.split('/')
    user_did = parts[4]
    profile_id = get_profile_id_from_user_did(user_did)
    post_id = parts[-1]
    return f"at://{profile_id}/app.bsky.feed.post/{post_id}"


def get_profile_id_from_user_did(user_did):
    profile = client.get_profile(user_did)
    return profile['did']

# Function to analyze sentiment of text
def analyze_text_sentiment(text):
    max_length = 512
    words = text.split()
    if len(words) > max_length:
        text = ' '.join(words[:max_length])

    model = models["Twitter RoBERTa"]
    sentiment = model(text)
    label = sentiment[0]['label']
    mapped_label = roberta_label_map.get(label, 'Unknown')
    score = sentiment[0]['score']
    return {'label': mapped_label, 'score': score}

# Register Bluesky login, post, and comment analysis functionality
def register_bluesky_callbacks(app):
    @app.callback(
        Output('bluesky-comment-results', 'children'),
        Input('analyze-comments-btn', 'n_clicks'),
        State('bluesky-post-url', 'value'),
        prevent_initial_call=True
    )
    def analyze_comments_of_post(n_clicks, post_url):
        if not post_url:
            return
        try:
            # Fetch thread and comments
            post_id = extract_post_id(post_url)
            post_thread = client.get_post_thread(post_id).thread
            comments = post_thread.replies or []

            if not comments:
                return html.Div(html.H5("No comments found for this post."))

            sentiment_results = []  # HTML elements for each comment/reply
            raw_sentiments = []     # {'label', 'score'} dicts

            # Helper to render a comment or reply with styled sentiment
            def record_and_render(text, parent_text=None):
                sent = analyze_text_sentiment(text)
                raw_sentiments.append(sent)
                color_map = {
                    'Positive': 'text-success',
                    'Neutral':  'text-secondary',
                    'Negative': 'text-danger'
                }
                sentiment_class = color_map.get(sent['label'], 'text-dark')

                sentiment_results.append(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                f"Reply to '{parent_text[:50]}...'" if parent_text else "Comment",
                                className="card-title"
                            ),
                            html.P(text, className="card-text"),
                            html.P(
                                f"Sentiment: {sent['label']} ({sent['score']:.2f})",
                                className=f"fw-bold {sentiment_class}",
                                style={'fontSize': '1.1rem'}
                            )
                        ]),
                        className="mb-3"
                    )
                )

            # Recursive reply analysis
            def analyze_replies(thread_post, parent_text):
                for reply in (thread_post.replies or []):
                    reply_text = getattr(reply.post.record, 'text', '') or ''
                    if not reply_text.strip():
                        continue
                    record_and_render(reply_text, parent_text)
                    analyze_replies(reply, reply_text)

            # Process top-level comments, skipping media-only
            for comment in comments:
                comment_text = getattr(comment.post.record, 'text', '') or ''
                if not comment_text.strip():
                    continue
                record_and_render(comment_text)
                analyze_replies(comment, comment_text)

            # Build summary figure with custom sentiment colors
            if raw_sentiments:
                counts = Counter([r['label'] for r in raw_sentiments])
                labels = list(counts.keys())
                values = list(counts.values())

                fig = px.pie(
                    names=labels,
                    values=values,
                    title='Overall Sentiment Distribution',
                )
                # Map labels to fixed colors: Positive=green, Neutral=grey, Negative=red
                graph_color_map = {'Positive': 'green', 'Neutral': 'grey', 'Negative': 'red'}
                fig.update_traces(
                    marker=dict(colors=[graph_color_map.get(lbl, 'grey') for lbl in labels])
                )

                # Add average confidence
                avg_conf = sum(r['score'] for r in raw_sentiments) / len(raw_sentiments)
                fig.add_annotation(
                    text=f"Avg. confidence: {avg_conf:.2f}",
                    x=0.5, y=-0.1, showarrow=False,
                    xref='paper', yref='paper'
                )

                summary_graph = dcc.Graph(
                    id='sentiment-summary-graph',
                    figure=fig,
                    style={'marginBottom': '2em'}
                )
            else:
                summary_graph = html.Div(html.H5("No comments to summarize."))

            # Return: summary graph, separator, then detailed cards
            return html.Div([
                dbc.Container([
                    summary_graph,
                    html.Hr(),
                    html.H4("Detailed Comment Analysis", className="mt-4 mb-2"),
                    *sentiment_results
                ], fluid=True)
            ])

        except Exception as e:
            return html.Div([
                html.H5("Error analyzing comments"),
                html.P(str(e))
            ])
