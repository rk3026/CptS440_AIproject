from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from logic.models import models, roberta_label_map, label_colors
from logic.model_handlers import *

def register_text_sentiment_callbacks(app):
    @app.callback(
        Output('text-analysis-results', 'children'),
        Input('analyze-btn', 'n_clicks'),
        State('input-text', 'value'),
        prevent_initial_call=True
    )
    def analyze_text(n_clicks, input_text):
        results = []

        for model_name, model in models.items():
            if isinstance(model, str):
                card = dbc.Card([
                    dbc.CardHeader(f"{model_name}"),
                    dbc.CardBody(html.P("Unsupported model"))
                ], className="mb-3")
                results.append(card)
                continue

            try:
                handler = model_handlers.get(model_name, GenericModelHandler())
                output = handler.analyze(model, input_text)

                # Multi-label models (GoEmotions, T5Emotions)
                if model_name in ["GoEmotions", "T5Emotions"]:
                    lines = []
                    for label, score in output:
                        color = label_colors.get(label.lower(), "black")
                        lines.append(
                            html.Span(
                                f"{label} ({score:.2f})",
                                style={
                                    "color": color,
                                    "fontWeight": "bold",
                                    "marginRight": "10px"
                                }
                            )
                        )

                    card = dbc.Card([
                        dbc.CardHeader(f"{model_name}"),
                        dbc.CardBody(html.P(lines))
                    ], className="mb-3")

                # Single-label models (like RoBERTa)
                else:
                    label, score = output
                    if model_name == "Twitter RoBERTa":
                        label = roberta_label_map.get(label, label)
                    color = label_colors.get(label.lower(), "black")

                    card = dbc.Card([
                        dbc.CardHeader(f"{model_name}"),
                        dbc.CardBody([
                            html.P(
                                f'Sentiment: {label}',
                                style={"color": color, "fontWeight": "bold"}
                            ),
                            html.P(f'Score: {score:.2f}')
                        ])
                    ], className="mb-3")

                results.append(card)

            except Exception as e:
                results.append(dbc.Card([
                    dbc.CardHeader(f"{model_name}"),
                    dbc.CardBody(html.P(f"Error - {str(e)}"))
                ], className="mb-3"))

        return html.Div(results)
