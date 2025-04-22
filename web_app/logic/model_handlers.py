'''
These are handler classes for analyzing text with a model
(each model has different methods to analyze, but they all have an analyze() function.
'''

from dash import Input, Output, State
from dash import html
from logic.models import models, roberta_label_map, goemotions_to_ekman
from collections import defaultdict

class BaseModelHandler:
    def analyze(self, model, input_text):
        raise NotImplementedError

class GoEmotionsHandler(BaseModelHandler):
    def analyze(self, model, input_text):
        prediction = model(input_text)
        emotions = [(item['label'], round(item['score'], 4)) for item in prediction[0]]

        from collections import defaultdict
        ekman_scores = defaultdict(float)
        for label, score in emotions:
            ekman_label = goemotions_to_ekman.get(label, "other")
            ekman_scores[ekman_label] += score

        sorted_ekman = sorted(ekman_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ekman

class T5EmotionsHandler(BaseModelHandler):
    def analyze(self, model, input_text):
        raw_output = model(input_text)[0]

        # Map the single label to Ekman emotions
        ekman_scores = defaultdict(float)
        ekman_label = goemotions_to_ekman.get(raw_output.strip(), "other")  # Mapping the output to Ekman emotion
        ekman_scores[ekman_label] += 1.0  # Assign score for the predicted label

        # Normalize scores (though it's just one label)
        total = sum(ekman_scores.values())
        if total > 0:
            for key in ekman_scores:
                ekman_scores[key] = round(ekman_scores[key] / total, 4)

        # Return sorted Ekman emotions (even though it's just one emotion)
        sorted_ekman = sorted(ekman_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ekman

class GenericModelHandler(BaseModelHandler):
    def analyze(self, model, input_text):
        prediction = model(input_text)
        return prediction[0]['label'], round(prediction[0]['score'], 4)
    
model_handlers = {
    "GoEmotions": GoEmotionsHandler(),
    "Twitter RoBERTa": GenericModelHandler(),
    "Yelp BERT": GenericModelHandler(),
    "T5Emotions": T5EmotionsHandler(),
    # "llama3": ...
}
