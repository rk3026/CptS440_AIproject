from dash import Input, Output, State
from dash import html
from logic.models import models, roberta_label_map, goemotions_to_ekman
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
                    ekman_str = ', '.join([f"{label} ({score})" for label, score in output])
                    results.append(html.Div([
                        html.H5(f'{model_name}:'),
                        html.P(f'Ekman Emotions: {ekman_str}')
                    ]))
                else:
                    label, score = output
                    if model_name == "Twitter RoBERTa":
                        label = roberta_label_map.get(label, label)
                    results.append(html.Div([
                        html.H5(f'{model_name}: {label}'),
                        html.P(f'Score: {score}')
                    ]))

            except Exception as e:
                results.append(html.Div([
                    html.H5(f'{model_name}: Error - {str(e)}')
                ]))

        return html.Div(results)
