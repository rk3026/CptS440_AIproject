from transformers import pipeline

# Define sentiment analysis models
models = {
    "Twitter RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
    "Yelp BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
    "GoEmotions": pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=True)
    #"llama3": "x",
    #"deepseek": "y",
    #"GPT-90": "z"
}

# Predefine the label mapping for Twitter RoBERTa
roberta_label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

goemotions_to_ekman = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "neutral",
    "caring": "joy",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}

