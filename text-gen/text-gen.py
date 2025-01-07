import transformers
import torch
from huggingface_hub import login
import os


login(token="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")

model_id = "meta-llama/Llama-3.2-1B-Instruct"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are currently in the United States looking for a sporty car with budget of $30,000."},
    {"role": "user", "content": "Choose a car and explain why."},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    input_text,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.75,
    top_p=0.75,
)

output_file = "generated_abstract.txt"
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        if outputs and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
            f.write(outputs[0]["generated_text"])
        elif outputs and "text" in outputs[0]:
            f.write(outputs[0]["text"])
        else:
            print("Unexpected pipeline output format:", outputs)
    print(f"Output successfully written to {output_file}")
except Exception as e:
    print(f"Error writing to file: {e}")

