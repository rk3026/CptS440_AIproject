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
        return sorted_ekman[:3]

class GenericModelHandler(BaseModelHandler):
    def analyze(self, model, input_text):
        prediction = model(input_text)
        return prediction[0]['label'], round(prediction[0]['score'], 4)
    
model_handlers = {
    "GoEmotions": GoEmotionsHandler(),
    "Twitter RoBERTa": GenericModelHandler(),
    "AnotherModel": GenericModelHandler(),
}

