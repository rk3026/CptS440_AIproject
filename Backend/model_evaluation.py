'''
This file evaluates all of the models used in the app.
'''

import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score, classification_report,
    mean_absolute_error, confusion_matrix
)

# ========================= MODEL LOADERS ========================= #

def load_t5_emotions_model(path="./models/t5-emotions"):
    model = T5ForConditionalGeneration.from_pretrained(path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    model.eval()
    return model, tokenizer

def load_goemotions_pipeline():
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def load_twitter_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def load_yelp_pipeline():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ========================= PREDICTORS ========================= #

def t5_predict(texts, label_names, model, tokenizer):
    preds = []
    for text in tqdm(texts, desc="T5 Predicting"):
        input_text = "classify sentiment: " + text
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, num_beams=1, eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        preds.append([label.strip() for label in decoded.split(',') if label.strip() in label_names])
    return preds

def goemotions_predict(texts, label_names, pipe, threshold=0.5):
    preds = []
    for text in tqdm(texts, desc="GoEmotions Predicting"):
        result = pipe(text)
        result = result[0] if isinstance(result, list) and isinstance(result[0], list) else result
        preds.append([item["label"] for item in result if item["score"] >= threshold])
    return preds

def twitter_predict(texts, pipe):
    label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    return [label_map[pipe(text)[0]["label"]] for text in tqdm(texts, desc="Twitter RoBERTa Predicting")]

def yelp_predict(texts, pipe):
    preds = []
    for text in tqdm(texts, desc="Yelp BERT Predicting"):
        try:
            result = pipe(text[:512])[0]  # truncate
            preds.append(int(result["label"][0]))
        except Exception as e:
            print(f"Error on text: {text[:100]}... -> {e}")
            preds.append(None)
    return preds

# ========================= EVALUATION ========================= #

def evaluate_multilabel(true_labels, pred_labels, label_names, name):
    mlb = MultiLabelBinarizer(classes=label_names)
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(pred_labels)
    print(f"\n----- {name} Evaluation -----")
    print(f"Micro F1: {f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Hamming Loss: {hamming_loss(y_true, y_pred):.4f}")

def evaluate_singlelabel(true, pred, model_name, labels=None, target_names=None):
    print(f"----- {model_name} Classification Report -----")
    print(classification_report(true, pred, labels=labels, target_names=target_names))

def evaluate_star_distance(true, pred, model_name):
    mae = mean_absolute_error(true, pred)
    print(f"----- {model_name} Star Distance -----")
    print(f"Mean Absolute Error (MAE): {mae:.4f} stars")

def evaluate_tolerance(true, pred, model_name):
    acc0 = sum(t == p for t, p in zip(true, pred)) / len(true)
    acc1 = sum(abs(t - p) <= 1 for t, p in zip(true, pred)) / len(true)
    print(f"----- {model_name} Tolerance Accuracy -----")
    print(f"Exact Match: {acc0:.4f}, Within ±1: {acc1:.4f}")

def plot_conf_matrix(true, pred, labels, model_name):
    cm = confusion_matrix(true, pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

# ========================= MAIN ========================= #

def main():
    # Load datasets
    goemo = load_dataset("go_emotions")
    test_data = goemo["test"]
    texts = [x["text"] for x in test_data]
    true_labels = [[test_data.features["labels"].feature.names[i] for i in x["labels"]] for x in test_data]
    label_names = test_data.features["labels"].feature.names

    # Eval T5Emotions
    t5_model, t5_tokenizer = load_t5_emotions_model()
    t5_preds = t5_predict(texts, label_names, t5_model, t5_tokenizer)
    evaluate_multilabel(true_labels, t5_preds, label_names, "T5Emotions")

    # Eval GoEmotions
    go_pipe = load_goemotions_pipeline()
    go_preds = goemotions_predict(texts, label_names, go_pipe)
    evaluate_multilabel(true_labels, go_preds, label_names, "GoEmotions")

    # Eval Twitter RoBERTa
    sst2 = load_dataset("glue", "sst2")["validation"]
    sst_texts = [x["sentence"] for x in sst2]
    sst_true = [x["label"] for x in sst2]
    tw_pipe = load_twitter_pipeline()
    sst_preds = twitter_predict(sst_texts, tw_pipe)
    evaluate_singlelabel(sst_true, sst_preds, "Twitter RoBERTa", labels=[0, 1], target_names=["Negative", "Positive"])

    # Eval Yelp BERT
    yelp_data = load_dataset("yelp_review_full", split="test[:1000]")
    yelp_texts = [x["text"] for x in yelp_data]
    yelp_true = [x["label"] + 1 for x in yelp_data]
    yelp_pipe = load_yelp_pipeline()
    yelp_preds = yelp_predict(yelp_texts, yelp_pipe)

    evaluate_singlelabel(yelp_true, yelp_preds, "Yelp BERT", labels=[1,2,3,4,5], target_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"])
    evaluate_star_distance(yelp_true, yelp_preds, "Yelp BERT")
    evaluate_tolerance(yelp_true, yelp_preds, "Yelp BERT")
    plot_conf_matrix(yelp_true, yelp_preds, labels=[1,2,3,4,5], model_name="Yelp BERT")

if __name__ == "__main__":
    main()
