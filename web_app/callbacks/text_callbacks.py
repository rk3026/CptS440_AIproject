from dash import Input, Output, State
from dash import html
from logic.models import models, roberta_label_map, label_colors
from collections import defaultdict
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
                results.append(html.Div([html.H5(f'{model_name}: Unsupported')]))
                continue

            try:
                handler = model_handlers.get(model_name, GenericModelHandler())
                output = handler.analyze(model, input_text)

                if model_name in ["GoEmotions", "T5Emotions"]:
                    # Colorize each Ekman label with bold
                    lines = []
                    for label, score in output:
                        color = label_colors.get(label.lower(), "black")
                        lines.append(
                            html.Span(
                                f"{label} ({score})",
                                style={
                                    "color": color,
                                    "fontWeight": "bold",
                                    "marginRight": "10px"
                                }
                            )
                        )

                    results.append(html.Div([
                        html.H5(f'{model_name}:'),
                        html.P(lines)
                    ]))
                else:
                    label, score = output
                    if model_name == "Twitter RoBERTa":
                        label = roberta_label_map.get(label, label)
                    color = label_colors.get(label.lower(), "black")
                    results.append(html.Div([
                        html.H5(f'{model_name}:'),
                        html.P(
                            f'Sentiment: {label}',
                            style={"color": color, "fontWeight": "bold"}
                        ),
                        html.P(f'Score: {score}')
                    ]))

            except Exception as e:
                results.append(html.Div([
                    html.H5(f'{model_name}: Error - {str(e)}')
                ]))

        return html.Div(results)
