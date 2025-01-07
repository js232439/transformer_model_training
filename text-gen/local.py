from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

login(token="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")

model_path = "/Users/js232439/.cache/huggingface/models--meta-llama--Llama-3.1-8B-Instruct"

generator = pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

input_text = (
    "Summarize this information into one sentence: "
    "Led and mentored a team of 20 associates, developing performance metrics and "
    "productivity goals, orchestrated distribution operations while maintaining safety protocols "
    "and efficiency standards, implemented process improvements that enhanced team productivity and workflow optimization."
)

outputs = generator(
    input_text,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.75,
    top_p=0.75,
)

output_file = "llama_gen.txt"
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(outputs[0]["generated_text"])
    print(f"Output successfully written to {output_file}")
except Exception as e:
    print(f"Error writing to file: {e}")
