from transformers import pipeline
from transformers import AutoConfig
import torch
from huggingface_hub import login
import os
from transformers import AutoTokenizer, AutoModelForCasualLM


login(token="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")

model_path = "/Users/js232439/.cache/huggingface/models--meta-llama--Llama-3.1-8B-Instruct"


pipeline = transformers.pipeline(
    "text-generation",
    model=mode_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a professional resume writer."},
    {"role": "user", "content": "Summarize this information on a resume into one sentence:: Led and mentored a team of 20 associates, developing performance metrics and productivity goals, Orchestrated distribution operations while maintaining safety protocols and efficiency standards, Implemented process improvements that enhanced team productivity and workflow optimization."},
]

tokenizer = AutoTokenizer.from_pretrained("/Users/js232439/.cache/huggingface/models--meta-llama--Llama-3.1-8B-Instruct")
model = AutoModelForCasualLM.from_pretrained("/Users/js232439/.cache/huggingface/models--meta-llama--Llama-3.1-8B-Instruct")

# terminators = [
#    pipeline.tokenizer.eos_token_id,
#    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#]

outputs = pipeline(
    messages,
    max_new_tokens=150,
    eos_token_id=tokenizer,
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

