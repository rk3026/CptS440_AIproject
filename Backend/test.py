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
print(google_dataset)