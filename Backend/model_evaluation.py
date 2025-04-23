from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, classification_report
from tqdm import tqdm
import torch

# ========== T5Emotions ==========
t5_model_path = "./models/t5-emotions"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model.eval()

def t5_predict(texts, label_names):
    preds = []
    for text in tqdm(texts, desc="T5 Predicting"):
        input_text = "classify sentiment: " + text
        input_ids = t5_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        with torch.no_grad():
            output_ids = t5_model.generate(input_ids, max_length=50, num_beams=1, eos_token_id=t5_tokenizer.eos_token_id)
        decoded = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        preds.append([label.strip() for label in decoded.split(',') if label.strip() in label_names])
    return preds

# ========== GoEmotions ==========
goemotions_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def goemotions_predict(texts, label_names, threshold=0.5):
    preds = []
    for text in tqdm(texts, desc="GoEmotions Predicting"):
        result = goemotions_pipeline(text)
        if isinstance(result, list) and isinstance(result[0], list):
            # result is [[{'label': ..., 'score': ...}, ...]]
            result = result[0]
        preds.append([item["label"] for item in result if item["score"] >= threshold])
    return preds

# ========== Twitter RoBERTa ==========
twitter_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
roberta_label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}  # 0=Neg, 1=Neutral, 2=Pos

def twitter_predict(texts):
    preds = []
    for text in tqdm(texts, desc="Twitter RoBERTa Predicting"):
        result = twitter_pipeline(text)[0]
        preds.append(roberta_label_map[result["label"]])
    return preds

# ========== Yelp BERT ==========
yelp_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def yelp_predict(texts):
    preds = []
    for text in tqdm(texts, desc="Yelp BERT Predicting"):
        # Truncate long inputs manually (optional - but usually safe to clip at ~450 words)
        truncated = text[:512]
        try:
            result = yelp_pipeline(truncated)[0]
            label = result["label"]
            # Extract just the number from label like "5 stars" -> 5
            preds.append(int(label[0]))
        except Exception as e:
            print(f"Error on text: {text[:100]}... -> {e}")
            preds.append(None)
    return preds


# ========== Evaluation ==========
def evaluate_multilabel(true_labels, pred_labels, label_names, name):
    mlb = MultiLabelBinarizer(classes=label_names)
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(pred_labels)

    print(f"\n----- {name} Evaluation on GoEmotions -----")
    print(f"Micro F1 Score: {f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Macro F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Subset Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Hamming Loss: {hamming_loss(y_true, y_pred):.4f}")

from sklearn.metrics import classification_report

def evaluate_singlelabel(true, pred, model_name, labels=None, target_names=None):
    print(f"----- {model_name} Classification Report -----")
    print(classification_report(true, pred, labels=labels, target_names=target_names))

from sklearn.metrics import mean_absolute_error

def evaluate_star_distance(true, pred, model_name):
    mae = mean_absolute_error(true, pred)
    print(f"----- {model_name} Star Distance Evaluation -----")
    print(f"Mean Absolute Error (MAE): {mae:.4f} stars")

def accuracy_within_tolerance(true, pred, tolerance=1):
    correct_within = sum(abs(t - p) <= tolerance for t, p in zip(true, pred))
    return correct_within / len(true)

def evaluate_tolerance(true, pred, model_name):
    acc1 = accuracy_within_tolerance(true, pred, tolerance=1)
    acc0 = accuracy_within_tolerance(true, pred, tolerance=0)
    print(f"----- {model_name} Tolerance Accuracy -----")
    print(f"Exact Match Accuracy: {acc0:.4f}")
    print(f"Accuracy Within ±1 Star: {acc1:.4f}")
    
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_conf_matrix(true, pred, labels, model_name):
    cm = confusion_matrix(true, pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

# ========== Run Evaluations ==========

# GoEmotions test data
goemo = load_dataset("go_emotions")
test_data = goemo["test"]
texts = [item["text"] for item in test_data]
true_labels = [[test_data.features["labels"].feature.names[i] for i in item["labels"]] for item in test_data]
label_names = test_data.features["labels"].feature.names

# 1. Evaluate T5
t5_preds = t5_predict(texts, label_names)
evaluate_multilabel(true_labels, t5_preds, label_names, "T5Emotions")

# 2. Evaluate GoEmotions
go_preds = goemotions_predict(texts, label_names)
evaluate_multilabel(true_labels, go_preds, label_names, "GoEmotions")

# 3. Twitter RoBERTa on SST2 (single-label sentiment)
sst2 = load_dataset("glue", "sst2")["validation"] # sst2 only has 0 for neg and 1 for pos
sst_texts = [x["sentence"] for x in sst2]
sst_true = [x["label"] for x in sst2]
sst_preds = twitter_predict(sst_texts)
evaluate_singlelabel(
    sst_true,
    sst_preds,
    model_name="Twitter RoBERTa",
    labels=[0, 1],
    target_names=["Negative", "Positive"]
)

# 4. Yelp BERT on Yelp sample data
yelp_sample = load_dataset("yelp_review_full", split="test[:1000]")
yelp_texts = [x["text"] for x in yelp_sample]
yelp_true_stars = [x["label"] + 1 for x in yelp_sample]
yelp_preds_stars = yelp_predict(yelp_texts)
print(yelp_true_stars)
print(yelp_preds_stars)

evaluate_singlelabel(
    yelp_true_stars,
    yelp_preds_stars,
    model_name="Yelp BERT",
    labels=[1, 2, 3, 4, 5],
    target_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
)

evaluate_star_distance(yelp_true_stars, yelp_preds_stars, "Yelp BERT")
evaluate_tolerance(yelp_true_stars, yelp_preds_stars, "Yelp BERT")
plot_conf_matrix(yelp_true_stars, yelp_preds_stars, labels=[1,2,3,4,5], model_name="Yelp BERT")
