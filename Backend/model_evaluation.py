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
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from datasets import load_dataset
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification # HuggingFace
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
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

# ========= PREDICTORS (Tests model prediction on given texts data) ========== #

def t5_predict(texts, label_names, model, tokenizer):
    preds = []
    for text in tqdm(texts, desc="T5 Predicting..."):
        input_text = "classify sentiment: " + text
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, num_beams=1, eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        preds.append([label.strip() for label in decoded.split(',') if label.strip() in label_names])
    return preds

def roberta_goemotions_predict(texts, label_names, pipe, threshold=0.5):
    preds = []
    for text in tqdm(texts, desc="Roberta GoEmotions Predicting..."):
        result = pipe(text)
        result = result[0] if isinstance(result, list) and isinstance(result[0], list) else result
        preds.append([item["label"] for item in result if item["score"] >= threshold])
    return preds

def twitter_predict(texts, pipe):
    label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    return [label_map[pipe(text)[0]["label"]] for text in tqdm(texts, desc="Twitter RoBERTa Predicting...")]


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

# ========== PREDICTORS WITH PROBABILITIES ========== #

# ========================= PREDICTORS ========================= #

def t5_predict_with_probs(texts, label_names, model, tokenizer):
    preds = []
    binary_probs = []

    for text in tqdm(texts, desc="T5 Predicting with probabilities..."):
        input_text = "classify sentiment: " + text
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, num_beams=1, eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_labels = [label.strip() for label in decoded.split(',') if label.strip() in label_names]
        preds.append(predicted_labels)
        binary_probs.append([1.0 if label in predicted_labels else 0.0 for label in label_names])

    return preds, binary_probs

def roberta_goemotions_predict_with_probs(texts, label_names, pipe, threshold=0.5, batch_size=16):
    preds = []
    probs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Roberta GoEmotions Predicting with probabilities..."):
        batch_texts = texts[i:i+batch_size]
        batch_results = pipe(batch_texts)

        for result in batch_results:
            result = result if isinstance(result, list) else [result]
            label_score_dict = {item["label"]: item["score"] for item in result}

            predicted_labels = [label for label, score in label_score_dict.items() if score >= threshold]
            preds.append(predicted_labels)

            full_prob_vector = [label_score_dict.get(label, 0.0) for label in label_names]
            probs.append(full_prob_vector)

    return preds, probs

def twitter_predict_with_probs(texts):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    preds = []
    probs = []

    for text in tqdm(texts, desc="Twitter Predicting with probabilities..."):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            softmax_probs = F.softmax(logits, dim=1).squeeze().numpy()
            probs.append(softmax_probs)
            preds.append(int(torch.argmax(logits)))

    return preds, probs

def yelp_predict_with_probs(texts, pipe):
    preds = []
    probs = []

    for text in tqdm(texts, desc="Yelp BERT Predicting with probabilities..."):
        try:
            result = pipe(text[:512])[0]
            preds.append(int(result["label"][0]))
            probs.append(result["score"])
        except Exception as e:
            print(f"Error on text: {text[:100]} -> {e}")
            preds.append(None)
            probs.append(None)

    return preds, probs

# ========================= EVALUATION HELPERS ========================= #

results = []

def record_result(model_name, metrics):
    for result in results:
        if result["Model"] == model_name:
            result.update(metrics)
            return
    results.append({"Model": model_name, **metrics})

def evaluate_multilabel(true_labels, pred_labels, label_names, name, probs=None):
    mlb = MultiLabelBinarizer(classes=label_names)
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(pred_labels)

    if probs is not None:
        probs = np.array(probs)
        if probs.shape[1] != len(label_names):
            raise ValueError(f"Shape mismatch: probs.shape={probs.shape}, expected {len(label_names)} labels")
        roc_auc_values = []
        for i in range(len(label_names)):
            if np.sum(y_true[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
                auc_value = auc(fpr, tpr)
                roc_auc_values.append(auc_value)
            else:
                roc_auc_values.append(np.nan)
        mean_auc = np.nanmean(roc_auc_values)
    else:
        mean_auc = None

    metrics = {
        "Micro F1": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Standard Deviation": np.std(y_true - y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Exact Match": (y_true == y_pred).all(axis=1).mean(),
        "Within ±1": np.mean([all(abs(t - p) <= 1 for t, p in zip(tr, pr)) for tr, pr in zip(y_true, y_pred)]),
        "Mean AUC": mean_auc
    }

    print_metrics(name, metrics)
    print_report(name, classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    record_result(name, metrics)
    if probs is not None:
        plot_roc_multilabel(y_true, np.array(probs), label_names, name)


def evaluate_singlelabel(true, pred, model_name, labels=None, target_names=None, probs=None):
    metrics = {
        "Micro F1": f1_score(true, pred, average='micro', zero_division=0),
        "Macro F1": f1_score(true, pred, average='macro', zero_division=0),
        "Accuracy": accuracy_score(true, pred),
        "Hamming Loss": hamming_loss(true, pred),
        "Standard Deviation": np.std(np.array(pred) - np.array(true)),
        "MAE": mean_absolute_error(true, pred),
        "Exact Match": np.mean(np.array(true) == np.array(pred)),
        "Within ±1": np.mean(np.abs(np.array(true) - np.array(pred)) <= 1),
    }

    # Only compute ROC AUC for binary classification
    if probs is not None and labels is not None and len(labels) == 2:
        probs_arr = np.array(probs)
        # for binary, pipeline probs might be 2-d (neg,pos) or 1-d (pos only)
        if probs_arr.ndim == 2 and probs_arr.shape[1] > 1:
            score = probs_arr[:, 1]
        else:
            score = probs_arr
        fpr, tpr, _ = roc_curve(true, score)
        metrics["ROC AUC"] = auc(fpr, tpr)

    print_metrics(model_name, metrics)
    print_report(
        model_name,
        classification_report(
            true,
            pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )
    )
    record_result(model_name, metrics)
    if probs is not None and labels is not None and len(labels) == 2:
        plot_roc_single(true, probs, model_name)


def plot_conf_matrix(true, pred, labels, model_name, evaluation_dir="evaluation"):
    os.makedirs(evaluation_dir, exist_ok=True)
    sanitized_model_name = model_name.replace('/', '_').replace(' ', '_').lower()

    if isinstance(true[0], list):
        mlb = MultiLabelBinarizer(classes=labels)
        y_true = mlb.fit_transform(true)
        y_pred = mlb.transform(pred)

        fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 3, 4))
        if len(labels) == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["0", "1"], yticklabels=["0", "1"])
            ax.set_title(labels[i])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
    else:
        cm = confusion_matrix(true, pred, labels=labels)
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

    filepath = os.path.join(evaluation_dir, f"{sanitized_model_name}_conf_matrix.pdf")
    plt.savefig(filepath)
    plt.close()

def export_report(results, evaluation_dir="evaluation"):
    os.makedirs(evaluation_dir, exist_ok=True)
    df = pd.DataFrame(results).fillna("–")
    filepath = os.path.join(evaluation_dir, "model_evaluation_report.pdf")
    with PdfPages(filepath) as pdf:
        fig, ax = plt.subplots(figsize=(14, len(df) * 0.6 + 1))
        ax.axis('off')
        table = pd.plotting.table(ax, df.round(4), loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
def plot_roc_multilabel(y_true, probs, label_names, model_name, evaluation_dir="evaluation"):
    """
    Plots one ROC curve per label plus the macro-average,
    and saves to `<evaluation_dir>/<model_name>_roc.pdf`.
    """
    from itertools import cycle
    os.makedirs(evaluation_dir, exist_ok=True)
    sanitized = model_name.replace('/', '_').replace(' ', '_').lower()
    filepath = os.path.join(evaluation_dir, f"{sanitized}_roc.pdf")

    # Compute per-label ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, label in enumerate(label_names):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC
    # First aggregate all FPR points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(label_names))]))
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(label_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(label_names)
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    # Plot per-label
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for i, (label, col) in enumerate(zip(label_names, colors)):
        plt.plot(fpr[i], tpr[i],
                 label=f"{label} (AUC = {roc_auc[i]:.2f})",
                 linewidth=1.5)

    # Plot macro
    plt.plot(fpr["macro"], tpr["macro"],
             label=f"Macro-avg (AUC = {roc_auc['macro']:.2f})",
             color='black', linestyle='--', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k:', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — ROC Curves")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_roc_single(true, probs, model_name, evaluation_dir="evaluation"):
    """
    Plots the binary ROC curve (pos vs. neg) and saves to `<model_name>_roc.pdf`.
    Accepts either:
      * probs shape (n_samples,) → interpreted as P(pos)
      * probs shape (n_samples,2) → takes column 1 as P(pos)
    """
    os.makedirs(evaluation_dir, exist_ok=True)
    sanitized = model_name.replace('/', '_').replace(' ', '_').lower()
    filepath = os.path.join(evaluation_dir, f"{sanitized}_roc.pdf")

    probs_arr = np.array(probs)
    if probs_arr.ndim == 2 and probs_arr.shape[1] > 1:
        score = probs_arr[:, 1]
    else:
        score = probs_arr

    fpr, tpr, _ = roc_curve(true, score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# ========================= PRINT HELPERS ========================= #

def print_section(title):
    print("\n" + "-" * 10 + f" {title} " + "-" * 10)

def print_metrics(name, metrics):
    print_section(f"{name} Evaluation")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

def print_report(name, report):
    print_section(f"{name} Classification Report")
    print(report)

# ========================= MAIN ========================= #

def main(max_samples=None):
    print("Loading GoEmotions dataset...")
    goemo = load_dataset("go_emotions")
    test_data = goemo["test"]
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    texts = [x["text"] for x in test_data]
    true_labels = [[test_data.features["labels"].feature.names[i] for i in x["labels"]] for x in test_data]
    label_names = test_data.features["labels"].feature.names

    print("Loading T5Emotions model...")
    t5_model, t5_tokenizer = load_t5_emotions_model()
    t5_preds, t5_probs = t5_predict_with_probs(texts, label_names, t5_model, t5_tokenizer)
    evaluate_multilabel(true_labels, t5_preds, label_names, "T5Emotions", probs=t5_probs)
    plot_conf_matrix(true_labels, t5_preds, label_names, "T5Emotions")

    print("Loading SamLowe GoEmotions model...")
    go_pipe = load_goemotions_pipeline()
    go_preds, go_probs = roberta_goemotions_predict_with_probs(texts, label_names, go_pipe)
    evaluate_multilabel(true_labels, go_preds, label_names, "SamLowe/roberta-base-go_emotions", probs=go_probs)
    plot_conf_matrix(true_labels, go_preds, label_names, "SamLowe/roberta-base-go_emotions")

    print("Loading SST-2 dataset...")
    sst2 = load_dataset("glue", "sst2")["validation"]
    if max_samples:
        sst2 = sst2.select(range(min(max_samples, len(sst2))))
    sst_texts = [x["sentence"] for x in sst2]
    sst_true = [x["label"] for x in sst2]

    print("Loading Twitter RoBERTa model...")
    tw_pipe = load_twitter_pipeline()
    sst_preds, sst_probs = twitter_predict_with_probs(sst_texts)
    evaluate_singlelabel(sst_true, sst_preds, "Twitter RoBERTa", labels=[0, 1], target_names=["Negative", "Positive"], probs=sst_probs)
    plot_conf_matrix(sst_true, sst_preds, labels=[0, 1], model_name="Twitter RoBERTa")

    print("Loading Yelp Reviews dataset...")
    yelp_data = load_dataset("yelp_review_full", split="test[:1000]")
    if max_samples:
        yelp_data = yelp_data.select(range(min(max_samples, len(yelp_data))))
    yelp_texts = [x["text"] for x in yelp_data]
    yelp_true = [x["label"] + 1 for x in yelp_data]

    print("Loading Yelp BERT model...")
    yelp_pipe = load_yelp_pipeline()
    yelp_preds, yelp_probs = yelp_predict_with_probs(yelp_texts, yelp_pipe)
    evaluate_singlelabel(yelp_true, yelp_preds, "Yelp BERT", labels=[1, 2, 3, 4, 5], target_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"], probs=yelp_probs)
    plot_conf_matrix(yelp_true, yelp_preds, labels=[1, 2, 3, 4, 5], model_name="Yelp BERT")

    export_report(results)

if __name__ == "__main__":
    main(max_samples=100)
