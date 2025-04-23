# callbacks/social_media_callbacks.py

from dash import Input, Output, State, callback_context, html
import dash
import dash_bootstrap_components as dbc
from atproto import Client
import os
from dotenv import load_dotenv
from logic.models import models, roberta_label_map, label_colors
from collections import Counter
import plotly.express as px


load_dotenv()
try:
    client = Client()
    client.login(os.getenv('BLUESKY_USER'), os.getenv('BLUESKY_APP_KEY'))
except Exception:
    client = None

def extract_post_id(post_url):
    parts = post_url.split('/')
    user_did = parts[4]
    profile_id = client.get_profile(user_did)['did']
    post_id = parts[-1]
    return f"at://{profile_id}/app.bsky.feed.post/{post_id}"

def analyze_text_sentiment(text):
    tokens = text.split()
    if len(tokens) > 512:
        text = ' '.join(tokens[:512])
    out = models["Twitter RoBERTa"](text)[0]
    label = roberta_label_map.get(out['label'], 'Unknown')
    return {'label': label, 'score': out['score']}

def register_bluesky_callbacks(app):

    @app.callback(
        Output('bluesky-comments-store',  'data'),
        Output('bluesky-sentiments-store', 'data'),
        Output('bluesky-interval',         'disabled'),
        Input('analyze-comments-btn', 'n_clicks'),
        Input('bluesky-interval',     'n_intervals'),
        State('bluesky-post-url',      'value'),
        State('bluesky-comments-store','data'),
        State('bluesky-sentiments-store','data'),
        prevent_initial_call=True
    )
    def batch_handler(btn_clicks, n_intervals,
                      post_url, comments_data,
                      sentiments_data):
        trig = callback_context.triggered[0]['prop_id'].split('.')[0]

        # 1) User clicked "Analyze Comments"
        if trig == 'analyze-comments-btn':
            if not post_url or not client:
                return dash.no_update, dash.no_update, True

            thread = client.get_post_thread(extract_post_id(post_url)).thread
            raw    = thread.replies or []

            flat = []
            def collect(node, parent):
                for r in (node.replies or []):
                    txt = getattr(r.post.record, 'text', '') or ''
                    if txt.strip():
                        flat.append({'text': txt, 'parent': parent})
                        collect(r, txt)

            for top in raw:
                txt = getattr(top.post.record, 'text', '') or ''
                if txt.strip():
                    flat.append({'text': txt, 'parent': None})
                    collect(top, txt)

            # reset both stores and enable interval
            return flat, [], False

        # 2) Interval tick: process next batch
        elif trig == 'bluesky-interval':
            if not comments_data:
                return dash.no_update, dash.no_update, True

            done_list = sentiments_data or []
            B  = 5
            i0 = len(done_list)
            i1 = min(i0 + B, len(comments_data))

            for entry in comments_data[i0:i1]:
                sent = analyze_text_sentiment(entry['text'])
                done_list.append({**entry, **sent})

            # disable interval if done
            return dash.no_update, done_list, (i1 >= len(comments_data))

        return dash.no_update, dash.no_update, True

    @app.callback(
        Output('sentiment-summary-graph', 'figure'),
        Input('bluesky-sentiments-store','data')
    )
    def update_pie(data):
        fig = px.pie(names=[], values=[], title='Overall Sentiment Distribution')
        if not data:
            return fig

        counts = Counter(d['label'] for d in data)
        labels, values = list(counts.keys()), list(counts.values())
        
        labels.sort()

        fig = px.pie(names=labels, values=values, title='Overall Sentiment Distribution')
        fig.update_traces(
            marker=dict(colors=[
                label_colors.get(label.lower(), 'lightblue') for label in labels
            ],
            line=dict(color='black', width=2)  # border between segments
            ),
            hovertemplate='Sentiment = %{label}<br>Count = %{value}<extra></extra>'
            
            #pull=[0.05 if v > 0 else 0 for v in values]  # slight separation
        )
        fig.update_layout(
            transition=dict(duration=500, easing='cubic-in-out'),
            uniformtext_minsize=12,
            uniformtext_mode='hide'
        )
        avg = sum(d['score'] for d in data) / len(data)
        fig.add_annotation(
            text=f"Average confidence: {avg:.2f}",
            x=0.5, y=-0.1, showarrow=False,
            xref='paper', yref='paper'
        )
        return fig

    @app.callback(
    Output('bluesky-comment-results','children'),
    Input('bluesky-sentiments-store','data'),
    State('bluesky-comment-results','children')
    )
    def append_cards(data, existing):
        # Clear old comments when starting a new analysis
        if data == []:
            return []

        existing = existing or []
        if not data:
            return existing

        start = len(existing)
        cards = []
        for d in data[start:]:
            label = d['label']
            color = label_colors.get(label.lower(), "lightblue")  # lowercase for GoEmotions too

            is_reply = bool(d['parent'])
            title = f"Reply to '{d['parent'][:50]}...'" if is_reply else "Comment"

            card = dbc.Card(
                dbc.CardBody([
                    html.H6(title, className='card-title'),
                    html.P(d['text'], className='card-text'),
                    html.P(
                        f"Sentiment: {label} ({d['score']:.2f})",
                        style={
                            'fontSize': '1.1rem',
                            'fontWeight': 'bold',
                            'color': color
                        }
                    )
                ]),
                className='mb-3',
                style={"marginLeft": "30px"} if is_reply else {}
            )
            cards.append(card)

        return existing + cards

    
    @app.callback(
        Output('bluesky-comment-count', 'children'),
        Input('bluesky-sentiments-store', 'data')
    )
    def update_comment_count(data):
        if not data:
            return "0"
        return f"{len(data)}"
