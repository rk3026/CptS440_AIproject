from dash import Input, Output, State
from dash import html
from logic.models import models, roberta_label_map, goemotions_to_ekman
from collections import defaultdict

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
            if isinstance(model, str):  # Skip unsupported models
                results.append(html.Div([html.H5(f'{model_name}: Unsupported')]))
                continue

            try:
                if model_name == "GoEmotions":
                    prediction = model(input_text)
                    emotions = [(item['label'], round(item['score'], 4)) for item in prediction[0]]  # <-- fixed

                    ekman_scores = defaultdict(float)
                    for label, score in emotions:
                        ekman_label = goemotions_to_ekman.get(label, "other")
                        ekman_scores[ekman_label] += score

                    sorted_ekman = sorted(ekman_scores.items(), key=lambda x: x[1], reverse=True)
                    top_ekman = sorted_ekman[:3]
                    ekman_str = ', '.join([f"{label} ({score})" for label, score in top_ekman])

                    results.append(html.Div([
                        html.H5(f'{model_name}:'),
                        html.P(f'Ekman Emotions: {ekman_str}')
                    ]))
                else:
                    prediction = model(input_text)
                    label = prediction[0]['label']
                    score = round(prediction[0]['score'], 4)

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
