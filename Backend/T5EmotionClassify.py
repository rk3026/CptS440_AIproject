from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def summarize_user_input():
    user_text = input("Enter the text you want to preform sentiment analysis:\n")

    padding_idx = 0
    eos_idx = 1
    max_seq_len = 512

    model_path = "./t5-emotions"  
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model.eval()
    

    #user input string for T5
    input_texts = [user_text]
    inputs = tokenizer(
    input_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    )

    # generate the text summary
    beam_size = 1
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_length=50,         
        num_beams=beam_size,
        eos_token_id=tokenizer.eos_token_id,
    )

    # decode the tokens
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("\n=== USER INPUT ===")
    print(f"User Input (original text): {user_text}")
    print("\n=== T5 Emotion Classification ===")
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    summarize_user_input()
