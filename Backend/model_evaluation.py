'''
This file evaluates all of the models used in the system with several metrics.

Evaluation Metrics:
- Accuracy: Proportion of correct predictions.
- F1 Score: Harmonic mean of precision and recall, useful for imbalanced datasets. Balances false positives and false negatives.
    Precision: Of all the predicted positives, how many were actually positive?
    Recall: Of all the actual positives, how many were correctly predicted?
- Micro F1: F1 score calculated globally by counting total true positives, false negatives, and false positives.
- Macro F1: F1 score calculated for each label, then averaged (treats all labels equally).
- Hamming Loss: Fraction of incorrect labels (used in multi-label classification).
- Standard Deviation: Measures variability between predicted and true labels.
- ROC Curve (Receiver Operating Characteristic): A plot of True Positive Rate vs. False Positive Rate across classification thresholds.
- AUC (Area Under the Curve): A single value summarizing the ROC curve's performance.
- MAE (Mean Absolute Error): Average absolute difference between predicted and true ratings.
- Tolerance Accuracy: Proportion of predictions within ±1 of the true rating.
'''

import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from datasets import load_dataset
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration # HuggingFace
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    confusion_matrix,
    roc_auc_score
)

import os
save_report_path = "evaluation"
os.makedirs(save_report_path, exist_ok=True)

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

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
def twitter_predict_with_probs(texts):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    preds = []
    probs = []
    for text in tqdm(texts, desc="Twitter Predicting"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            softmax_probs = F.softmax(logits, dim=1).squeeze().numpy()
            probs.append(softmax_probs)
            preds.append(int(torch.argmax(logits)))
    return preds, probs


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

results = []

def record_result(model_name, metrics):
    for result in results:
        if result["Model"] == model_name:
            result.update(metrics)
            return
    results.append({"Model": model_name, **metrics})

def evaluate_multilabel(true_labels, pred_labels, label_names, name):
    mlb = MultiLabelBinarizer(classes=label_names)
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(pred_labels)
    metrics = {
        "Micro F1": f1_score(y_true, y_pred, average='micro'),
        "Macro F1": f1_score(y_true, y_pred, average='macro'),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred)
    }
    print(f"\n----- {name} Evaluation -----")
    print(metrics)
    print(f"\n----- {name} Classification Report -----")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    record_result(name, metrics)

def evaluate_singlelabel(true, pred, model_name, labels=None, target_names=None, probs=None):
    metrics = {
        "Micro F1": f1_score(true, pred, average='micro'),
        "Macro F1": f1_score(true, pred, average='macro'),
        "Accuracy": accuracy_score(true, pred),
        "Hamming Loss": hamming_loss(true, pred),
        "Standard Deviation": np.std(np.array(pred) - np.array(true))
    }

    # AUC (needs probs and binarized labels)
    if probs is not None and labels is not None and len(set(true)) > 1:
        y_true_bin = label_binarize(true, classes=labels)
        try:
            auc = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
            metrics["AUC"] = auc
        except ValueError as e:
            print(f"Skipping AUC for {model_name}: {e}")
            metrics["AUC"] = None
    else:
        metrics["AUC"] = None

    print(f"\n----- {model_name} Evaluation -----")
    print(metrics)
    print(f"----- {model_name} Classification Report -----")
    print(classification_report(true, pred, labels=labels, target_names=target_names))
    record_result(model_name, metrics)

def evaluate_star_distance(true, pred, model_name):
    mae = mean_absolute_error(true, pred)
    print(f"{model_name} MAE: {mae}")
    record_result(model_name, {"MAE": mae})

def evaluate_tolerance(true, pred, model_name):
    acc0 = sum(t == p for t, p in zip(true, pred)) / len(true)
    acc1 = sum(abs(t - p) <= 1 for t, p in zip(true, pred)) / len(true)
    print("Tolerance Accuracy")
    print(f"{model_name} Exact Match: {acc0}")
    print(f"{model_name} Within ±1 star: {acc1}")
    record_result(model_name, {"Exact Match": acc0, "Within ±1": acc1})

def plot_conf_matrix(true, pred, labels, model_name, evaluation_dir="evaluation"):
    os.makedirs(evaluation_dir, exist_ok=True)
    cm = confusion_matrix(true, pred, labels=labels)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    filename = f"{model_name.replace(' ', '_').lower()}_conf_matrix.pdf"
    filepath = os.path.join(evaluation_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
def plot_roc_curve(y_true, y_probs, model_name, labels, save_dir="evaluation"):
    y_true_bin = label_binarize(y_true, classes=labels)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], [p[i] for p in y_probs])
        plt.plot(fpr, tpr, label=f"{label} (class {i})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.pdf"))
    plt.close()


def export_report(results, evaluation_dir="evaluation"):
    os.makedirs(evaluation_dir, exist_ok=True)
    df = pd.DataFrame(results).fillna("–")
    filename = "model_evaluation_report.pdf"
    filepath = os.path.join(evaluation_dir, filename)

    with PdfPages(filepath) as pdf:
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1))
        ax.axis("off")
        table = pd.plotting.table(ax, df.round(4), loc="center", cellLoc='center', colWidths=[0.2] * len(df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

# ========================= MAIN ========================= #

def main(max_samples=None):
    #----------Evaluate T5Emotions------------#
    # Load GoEmotions test data
    print("Loading GoEmotions Data...")
    goemo = load_dataset("go_emotions")
    print("GoEmotions Data Loaded!")
    test_data = goemo["test"]
    # Reduce the amount of data to test on if specified:
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    texts = [x["text"] for x in test_data]
    true_labels = [[test_data.features["labels"].feature.names[i] for i in x["labels"]] for x in test_data]
    label_names = test_data.features["labels"].feature.names
    # Print an example row:
    print(f"Row 0 text: {texts[0]}")
    print(f"Row 0 actual/truth label: {test_data.features["labels"].feature.names[0]}")

    # T5Emotions (our custom fine-tuned model)
    print("Loading T5Emotions Model...")
    t5_model, t5_tokenizer = load_t5_emotions_model()
    print("T5Emotions Model Loaded!")
    t5_preds = t5_predict(texts, label_names, t5_model, t5_tokenizer)
    print("Evaluating T5Emotions...")
    evaluate_multilabel(true_labels, t5_preds, label_names, "T5Emotions")
    print("T5Emotions Evaluation Complete!")
    #-------------------------------------------------------#

    #----------Evaluate roberta-base-go_emotions------------#
    # SamLowe/roberta-base-go_emotions
    print("Loading SamLowe/roberta-base-go_emotions Model...")
    go_pipe = load_goemotions_pipeline()
    print("SamLowe/roberta-base-go_emotions Model Loaded!")
    go_preds = goemotions_predict(texts, label_names, go_pipe)
    print("Evaluating SamLowe/roberta-base-go_emotions...")
    evaluate_multilabel(true_labels, go_preds, label_names, "SamLowe/roberta-base-go_emotions")
    print("SamLowe/roberta-base-go_emotions Evaluation Complete!")
    #-------------------------------------------------------#

    #----------Evaluate Twitter RoBERTa------------#
    # Twitter RoBERTa
    # Load test data:
    print("Loading sst2 Test Data...")
    sst2 = load_dataset("glue", "sst2")["validation"]
    print("sst2 Test Data Loaded!")
    if max_samples:
        sst2 = sst2.select(range(min(max_samples, len(sst2))))
    sst_texts = [x["sentence"] for x in sst2]
    sst_true = [x["label"] for x in sst2]
    # Load model:
    print("Loading Twitter RoBERTa Model...")
    tw_pipe = load_twitter_pipeline()
    print("Twitter RoBERTa Model Loaded!")
    print("Beginning Testing Model on Test Data...")
    
    sst_preds = twitter_predict(sst_texts, tw_pipe)
    evaluate_singlelabel(sst_true, sst_preds, "Twitter RoBERTa", labels=[0, 1], target_names=["Negative", "Positive"])
    
    '''
    sst_preds, sst_probs = twitter_predict_with_probs(sst_texts)
    sst_probs = [[1 - p, p] for p in sst_probs]
    print("Model Testing Finished!")
    print("Evaluating TwitterRoBERTa Performance...")
    evaluate_singlelabel(sst_true, sst_preds, "Twitter RoBERTa",
                        labels=[0, 1], target_names=["Negative", "Positive"], probs=sst_probs)
    plot_roc_curve(sst_true, sst_probs, "Twitter RoBERTa", labels=[0, 1])
    '''
    print("Twitter RoBERTa Evaluation Finished!")
    #-------------------------------------------------------#

    #----------Evaluate Yelp BERT------------#
    # Yelp BERT
    print("Loading Yelp Reviews Dataset...")
    yelp_data = load_dataset("yelp_review_full", split="test[:1000]")
    if max_samples:
        yelp_data = yelp_data.select(range(min(max_samples, len(yelp_data))))
    yelp_texts = [x["text"] for x in yelp_data]
    yelp_true = [x["label"] + 1 for x in yelp_data]
    print("Yelp Reviews Dataset Loaded!")
    print("Loading Yelp BERT Model...")
    yelp_pipe = load_yelp_pipeline()
    print("Yelp BERT Model Loaded!")
    print("Beginning Testing Model on Test Data...")
    yelp_preds = yelp_predict(yelp_texts, yelp_pipe)
    print("Model Testing Finished!")

    print("Beginning Yelp BERT Performance Evaluation...")
    evaluate_singlelabel(yelp_true, yelp_preds, "Yelp BERT", labels=[1, 2, 3, 4, 5],
                         target_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"])
    evaluate_star_distance(yelp_true, yelp_preds, "Yelp BERT")
    evaluate_tolerance(yelp_true, yelp_preds, "Yelp BERT")
    plot_conf_matrix(yelp_true, yelp_preds, labels=[1, 2, 3, 4, 5], model_name="Yelp BERT")
    print("Yelp BERT Evaluation Finished!")
    #-------------------------------------------------------#

    export_report(results)


if __name__ == "__main__":
    main(max_samples=100)
