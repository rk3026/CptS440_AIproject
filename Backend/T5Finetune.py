#this model was trained on go_emotions which is dataset from google 
# it was trained on gpu from google a100 from colab pro
#https://huggingface.co/docs/
#
from datasets import load_dataset, Features, Sequence, Value
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

# Load the multi-label GoEmotions dataset (default configuration)
google_dataset = load_dataset("go_emotions")

# Retrieve and store label names (needed for converting indices to strings later)
label_names = google_dataset["train"].features["labels"].feature.names

# Ran into issues because the T5 tokenizer converts label strings into token IDs
# and these token IDs can be much larger than the original class label range (0 to 27) which is used in the dataset
# Since the original 'labels' column is defined as a ClassLabel,
# it only accepts integers within the range 0 to 27. To fix the issue, we recast the dataset 
# so that T5 can assign its tokenID without corrupting the original requiremen 0 to 27 range
#not sure if roberta or bert will have similar issues
new_features = google_dataset["train"].features.copy()
new_features["labels"] = Sequence(Value("int64"))
google_dataset = google_dataset.cast(new_features)

# Initialize the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Preprocessing function to create input and target text, then tokenize them
def preprocess(batch):
    # Create a prompt for each example (you can adjust the wording as needed)
    input_texts = ["classify sentiment: " + text for text in batch["text"]]
    
    # For multi-label targets, convert list of indices to comma-separated label names
    target_texts = []
    for label_list in batch["labels"]:
        label_strs = [label_names[label_id] for label_id in label_list]
        target_texts.append(", ".join(label_strs))
    
    # Tokenize inputs
    model_inputs = tokenizer(
        input_texts,
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize target texts
    labels = tokenizer(
        target_texts,
        max_length=8,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map the preprocessing function over the dataset and remove the original columns
tokenized_dataset = google_dataset.map(
    preprocess,
    batched=True,
    remove_columns=google_dataset["train"].column_names,
    desc="Tokenizing dataset"
)

# the data collator used in huggingfaace understands the structure required for sequence-to-sequence tasks
# so it handles the formatting for endcoding and decoding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Set the training arguments following huggingface
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-emotion-model",      # directory to save the model so i can download from colab
    eval_strategy="epoch",                # use eval_strategy 
    learning_rate=3e-4,                   # learning rate for the model
    per_device_train_batch_size=16,       # training batch size per device
    per_device_eval_batch_size=16,        # evaluation batch size per device
    num_train_epochs=3,                   # total number of epochs
    weight_decay=0.01,                    # weight decay to minimize overfitting
    save_total_limit=2,                   # limit to the 2 most recent checkpoints
    logging_dir="./logs",                 # directory to store logs
    predict_with_generate=True,           # use generate() method for prediction
    report_to=["tensorboard"],            # disable wandb by reporting only to TensorBoard ecnoutered api key for wandb
    run_name="t5_emotion_run"             # optional: set a custom run name to avoid wandb warnings
)

# Initialize the trainer.
#  pass the data collator instead of the tokenizer.
# from hugging face:
#Data collators are objects that will form a batch by using a list of dataset elements as input.
# These elements are of the same type as the elements of train_dataset or eval_dataset.
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./t5-emotions")
tokenizer.save_pretrained("./t5-emotions")