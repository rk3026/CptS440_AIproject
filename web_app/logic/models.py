from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the local T5-emotions model
t5_model_path = "./models/t5-emotions"  
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model.eval()

def t5_emotion_classification(texts):
    inputs = t5_tokenizer(
        ["classify sentiment: " + t for t in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=50,
            num_beams=1,
            eos_token_id=t5_tokenizer.eos_token_id
        )

    predictions = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions


# Define sentiment analysis models
models = {
    "Twitter RoBERTa": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
    "Yelp BERT": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
    "GoEmotions": pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None),
    "T5Emotions": t5_emotion_classification  # no pipeline here, as it is local (not HuggingFace)
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

# Label maps for GoEmotion to ekman emotions
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

label_colors = {
    # RoBERTa Sentiment Labels
    "positive": "green",
    "neutral": "grey",
    "negative": "red",

    # Ekman Emotions
    "joy": "#B8860B",  # DarkGoldenrod
    "sadness": "blue",
    "anger": "darkred",
    "fear": "purple",
    "surprise": "orange",
    "disgust": "olive",
    # "neutral": "grey",  # lowercase for GoEmotions

    # Fallback
    "other": "lightblue"
}
