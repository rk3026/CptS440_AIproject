from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

#google emotions data set
goemo = load_dataset("go_emotions")
#runnning metrics on the test data from the dataset
# was trained on the training data
test_data = goemo["test"]

# this gets the lable names- so all the emotions
label_names = test_data.features["labels"].feature.names


#path to the fine tuned model
model_path = "./t5-emotions" 

#this converts the text to token ids for the t5 model (thats just how it works in this model)
tokenizer = T5Tokenizer.from_pretrained(model_path)

#load the model
model = T5ForConditionalGeneration.from_pretrained(model_path)

#use evalutate because we're evaluting the model
model.eval()

def predict_emotions(text):
    #this is the task for the model so "classify sentiment" is the prompt for the model
    #this lets the model know what you want to do, for example it can also preform summarization among others
    input_text = "classify sentiment: " + text
    # tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
    # this will generate the sentiment predictions without gradients
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_beams=1, eos_token_id=tokenizer.eos_token_id)
    # gotta decode the tokens back into text
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #return the list of predictions
    return [label.strip() for label in decoded.split(',') if label.strip() in label_names]

# lists of predictions and true labels
predictions = []
true_labels = []

#iterate through all the test data and collections the preditions and truth
for item in tqdm(test_data):
    preds = predict_emotions(item["text"])
    predictions.append(preds)
    true_labels.append([label_names[i] for i in item["labels"]])

# convert the lists into binary vectors for the sklearn metrics
mlb = MultiLabelBinarizer(classes=label_names)
y_true = mlb.fit_transform(true_labels)
y_pred = mlb.transform(predictions)

# Metrics
print("\n-------------Evaluation Metrics on Google_Emotions-------------")
print("\nTest set was used for metrics. \nThe model was fine tuned using the training set ")
#micro F1 score calculates all true postives, false postives and false negatives
# so essnetially this how we see how well the model can predict
# higher is better, how accurate is it
micro = f1_score(y_true, y_pred, average='micro')
print(f"Micro F1 Score: {micro:.4f} \nMicro F1: The model was accurate {micro * 100:.2f}% of the time across all emotion predictions.")
# This lets us see how well the model predicts each lable 
# higher is better, how well does it predict each emotion
macro = f1_score(y_true, y_pred, average='macro')
print(f"Macro F1 Score: {macro:.4f} \nMacro F1: On average, the model was {macro * 100:.2f}% accurate across each individual emotion.")

# this metric is for identifying if the model got every emoiton correct
# higher is better, model predicts all the correct emotions exactly for the given sentence
subset_acc = accuracy_score(y_true, y_pred)
print(f"Subset Accuracy: {subset_acc:.4f} \nSubset Accuracy: The model made perfect emotion predictions for {subset_acc * 100:.2f}% of sentences.")
# this metrics accounts for how many mistakes were made
# lower is better, how many emotions are being predicted incorrectly?
hamming = hamming_loss(y_true, y_pred)
print(f"Hamming Loss: {hamming:.4f} \nHamming Loss: On average, the model made mistakes on {hamming * 100:.2f}% of possible emotion labels.")