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

# Test cases from ross
test_cases_1 = [
    "This movie was lit 🔥🔥!",
    "I love this song so much 😍",
    "I’m having a great time, everyone is so friendly!",
    "Such a great experience, I highly recommend it!",
    "Ugh, I hate waiting for this!",
    "Why is this place always so crowded? 😩",
    "I can't believe how bad the service was!",
    "This place sucks!",
    "It's just okay, not great but not terrible.",
    "I don't know what to feel about it. 🤔"
]

test_cases_2 = [
    "The food was fantastic, but the service was slow.",
    "This place was terrible. The food was cold and the staff was rude.",
    "I had a wonderful experience here! The staff was friendly, and the food was amazing!",
    "The ambiance was nice, but the food wasn't great. I expected better.",
    "Absolutely awful. Never coming back.",
    "Loved the decor and the staff, but the food was too spicy for me.",
    "This restaurant is incredible! One of the best meals I've ever had!",
    "Not worth the price. Very disappointing.",
    "Great food, great service. Highly recommend!",
    "The waiter was nice, but the food was just okay."
]

## Set up CSV logging. Doesnt work well yet. ##

# Set up the file path for storing results
results_file = 'sentiment_analysis_results.csv'

# Ensure the CSV exists (create it if not)
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["test_case", "model", "sentiment", "score"])

# Function to log results to the CSV file if the entry does not exist yet
def log_results(test_case, model_name, sentiment, score):
    # If file exists, read it and check if the combination exists
    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            if ((df['test_case'] == test_case) & (df['model'] == model_name)).any():
                return
        except Exception as e:
            pass  # If reading fails, simply proceed to log
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([test_case, model_name, sentiment, score])

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
    
    # Dropdown to select previous test case results
    dcc.Dropdown(
        id="previous-results-dropdown",
        options=[{'label': f"Test Case Set {i+1}", 'value': i+1} for i in range(2)],
        placeholder="Select previous test case results"
    ),
    
    # Graph to show results
    dcc.Graph(id="results-plot")
])

# Callback to update the plot based on the selected models and input text or test cases
@app.callback(
    Output("results-plot", "figure"),
    Output("previous-results-dropdown", "options"),
    Input("analyze-btn", "n_clicks"),
    Input("input-text", "value"),
    Input("model-selection", "value"),
    Input("previous-results-dropdown", "value")
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
            log_results(text, model_name, sentiment, score)
        
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
    
    # When a previous test case set is selected
    if selected_previous:
        test_cases = test_cases_1 if selected_previous == 1 else test_cases_2
        results_list = []
        for idx, case in enumerate(test_cases, start=1):
            for model_name, model in models.items():
                res = model(case)
                raw_label = res[0]['label']
                score = res[0]['score']
                sentiment = classify_sentiment(raw_label)
                results_list.append({
                    "test_case": f"Test Case {idx}",
                    "model": model_name,
                    "sentiment": sentiment,
                    "score": score
                })
                log_results(case, model_name, sentiment, score)
        
        # Convert the results list into a DataFrame
        df = pd.DataFrame(results_list)
        # Create a grouped bar chart for the test cases
        fig = go.Figure()
        for model_name in df["model"].unique():
            df_model = df[df["model"] == model_name]
            fig.add_trace(go.Bar(
                x=df_model["test_case"],
                y=df_model["score"],
                text=[f'{s} ({sc:.2f})' for s, sc in zip(df_model["sentiment"], df_model["score"])],
                textposition='auto',
                name=model_name
            ))
        fig.update_layout(
            title="Sentiment Analysis for Test Case Set",
            xaxis_title="Test Case",
            yaxis_title="Confidence Score",
            barmode='group'
        )
        return fig, dropdown_options
    
    # Default empty figure
    return go.Figure(), dropdown_options

# Run the server
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4200)



# We would need some sort of normalization algorithm to solve the differences between how modles are classifying things.