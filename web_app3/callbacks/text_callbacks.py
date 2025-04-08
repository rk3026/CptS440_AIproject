from dash import Input, Output, State
from logic.models import models, roberta_label_map
from dash import html

def register_text_sentiment_callbacks(app):
    # Callback for Text Analysis
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

            try:
                prediction = model(input_text)
                label = prediction[0]['label']
                score = round(prediction[0]['score'], 4)

                # Use predefined label map for Twitter RoBERTa
                if model_name == "Twitter RoBERTa":
                    label = roberta_label_map.get(label, label)

                results.append(html.Div([html.H5(f'{model_name}: {label}'), html.P(f'Score: {score}')]))
            except Exception as e:
                results.append(html.Div([html.H5(f'{model_name}: Error - {str(e)}')]))

        return html.Div(results)
