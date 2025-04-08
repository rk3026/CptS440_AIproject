import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from transformers import pipeline
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import csv
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Define sentiment analysis models https://huggingface.co/docs/transformers/main_classes/pipelines 
models = {
    "Twitter RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
    "Yelp BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
    #"xlnet-base-cased": pipeline("sentiment-analysis", model="xlnet-base-cased"),
    #"t5-small": pipeline("text2text-generation", model="t5-small")
}

# Function to classify sentiment into positive, negative, or neutral
def classify_sentiment(label):
    label_lower = label.lower()
    if "star" in label_lower:
        # Extract the first character as the star rating (e.g., "5 stars")
        try:
            num = int(label_lower[0])
        except:
            num = 3
        if num <= 2:
            return "negative"
        elif num == 3:
            return "neutral"
        else:
            return "positive"
    else:
        if "positive" in label_lower:
            return "positive"
        elif "negative" in label_lower:
            return "negative"
        elif "neutral" in label_lower:
            return "neutral"
        else:
            return "neutral"

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Sentiment Analysis App"),
    
    # Text input for user to enter text
    dcc.Textarea(
        id="input-text",
        value="Enter text for sentiment analysis",
        style={'width': '100%', 'height': 100}
    ),
    
    # Checkboxes for selecting models
    dcc.Checklist(
        id="model-selection",
        options=[{'label': model, 'value': model} for model in models.keys()],
        value=["Twitter RoBERTa", "Yelp BERT"],  # Default models
        style={'margin-top': 20}
    ),
    
    # Button to trigger analysis
    html.Button("Analyze", id="analyze-btn", n_clicks=0),
])

# Callback to update the plot based on the selected models and input text or test cases
@app.callback(
    Input("analyze-btn", "n_clicks"),
    Input("input-text", "value"),
    Input("model-selection", "value"),
)
def update_plot(n_clicks, text, selected_models, selected_previous):
    # Default options for the dropdown
    dropdown_options = [{'label': f"Test Case Set {i+1}", 'value': i+1} for i in range(2)]
    
    # When custom text is analyzed (button clicked)
    if n_clicks > 0 and text and selected_models:
        results_list = []
        for model_name in selected_models:
            res = models[model_name](text)
            raw_label = res[0]['label']
            score = res[0]['score']
            sentiment = classify_sentiment(raw_label)
            results_list.append({
                "test_case": "Input Text",
                "model": model_name,
                "sentiment": sentiment,
                "score": score
            })
        
        # Create a bar chart: x-axis is model name, y-axis is the confidence score
        fig = go.Figure()
        for entry in results_list:
            fig.add_trace(go.Bar(
                x=[entry["model"]],
                y=[entry["score"]],
                text=f'{entry["sentiment"]} ({entry["score"]:.2f})',
                textposition='auto',
                name=entry["model"]
            ))
        fig.update_layout(
            title="Sentiment Analysis for Input Text",
            xaxis_title="Model",
            yaxis_title="Confidence Score"
        )
        return fig, dropdown_options

# Run the server
if __name__ == "__main__":
    app.run(debug=True, port=4200)



# We would need some sort of normalization algorithm to solve the differences between how modles are classifying things.