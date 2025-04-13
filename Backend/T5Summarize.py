from torchtext.models import T5_BASE_GENERATION
from torchtext.prototype.generate import GenerationUtils
from torchtext.models import T5Transform

def summarize_user_input():
    # 1) Prompt user for text
    user_text = input("Enter the text you want to summarize:\n")

    # 2) Set up T5
    padding_idx = 0
    eos_idx = 1
    max_seq_len = 512

    t5_base = T5_BASE_GENERATION
    transform = t5_base.transform()  
    model = t5_base.get_model()
    model.eval()
    sequence_generator = GenerationUtils(model)

    # set up the task for t5 so far have tried sst2 for classificaiton of pos/neg
    task_prefix = "summarize: "

    #user input string for T5
    input_with_prefix = [task_prefix + user_text]

    # tokenize the input
    model_input = transform(input_with_prefix)

    # generate the text summary
    beam_size = 1
    model_output = sequence_generator.generate(
        model_input,
        eos_idx=eos_idx,
        num_beams=beam_size
    )

    # Decode tokens -> strings
    summary_output = transform.decode(model_output.tolist())

    # print the result
    print("\n=== USER INPUT ===")
    print(f"User Input (original text): {user_text}")
    print("\n=== T5 SUMMARY ===")
    print(summary_output[0])
    print("-" * 50)

if __name__ == "__main__":
    summarize_user_input()
