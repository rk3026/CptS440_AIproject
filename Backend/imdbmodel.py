from torchtext.models import T5_BASE_GENERATION
from torchtext.prototype.generate import GenerationUtils
from functools import partial
from torchtext.datasets import IMDB
from functools import partial
from torch.utils.data import DataLoader
from torchtext.datasets import CNNDM
from torchtext.models import T5Transform

padding_idx = 0
eos_idx = 1
max_seq_len = 512
t5_sp_model_path = "https://download.pytorch.org/models/text/t5_tokenizer_base.model"

transform = T5Transform(
    sp_model_path=t5_sp_model_path,
    max_seq_len=max_seq_len,
    eos_idx=eos_idx,
    padding_idx=padding_idx,
)
t5_base = T5_BASE_GENERATION
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()
sequence_generator = GenerationUtils(model)

cnndm_batch_size = 5
cnndm_datapipe = CNNDM(split="test")
task = "summarize"


def apply_prefix(task, x):
    return f"{task}: " + x[0], x[1]


cnndm_datapipe = cnndm_datapipe.map(partial(apply_prefix, task))
cnndm_datapipe = cnndm_datapipe.batch(cnndm_batch_size)
cnndm_datapipe = cnndm_datapipe.rows2columnar(["article", "abstract"])
cnndm_dataloader = DataLoader(cnndm_datapipe, shuffle=True, batch_size=None)

imdb_batch_size = 3
imdb_datapipe = IMDB(split="test")
task = "sst2 sentence"
labels = {"1": "negative", "2": "positive"}


def process_labels(labels, x):
    return x[1], labels[str(x[0])]


imdb_datapipe = imdb_datapipe.map(partial(process_labels, labels))
imdb_datapipe = imdb_datapipe.map(partial(apply_prefix, task))
imdb_datapipe = imdb_datapipe.batch(imdb_batch_size)
imdb_datapipe = imdb_datapipe.rows2columnar(["text", "label"])
imdb_dataloader = DataLoader(imdb_datapipe, batch_size=None)

batch = next(iter(imdb_dataloader))
input_text = batch["text"]
target = batch["label"]
beam_size = 1

model_input = transform(input_text)
model_output = sequence_generator.generate(model_input, eos_idx=eos_idx, num_beams=beam_size)
output_text = transform.decode(model_output.tolist())

for i in range(imdb_batch_size):
    print(f"Example {i+1}:\n")
    print(f"input_text: {input_text[i]}\n")
    print(f"prediction: {output_text[i]}\n")
    print(f"target: {target[i]}\n\n")