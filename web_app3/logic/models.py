from transformers import pipeline

# Define sentiment analysis models
models = {
    "Twitter RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
    "Yelp BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
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
