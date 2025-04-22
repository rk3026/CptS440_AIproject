from dash import Dash
import dash_bootstrap_components as dbc

from layout.main_layout import get_main_layout
from callbacks.yelp_callbacks import register_yelp_callbacks
from callbacks.text_callbacks import register_text_sentiment_callbacks
from callbacks.social_media_callbacks import register_bluesky_callbacks

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Sentiment Analysis Dashboard",
    suppress_callback_exceptions=True
)
server = app.server

# Load layout
app.layout = get_main_layout()

# Register all callbacks
register_yelp_callbacks(app)
register_text_sentiment_callbacks(app)
register_bluesky_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=4200)
