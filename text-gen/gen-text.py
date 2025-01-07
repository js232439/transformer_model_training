from airllm import AirLLM
from transformers import AutoTokenizer

# Load the LLaMA 3.3 model and tokenizer
model_name = "/Users/js232439/javi-sandbox/text-gen/Llama-3.3-70B-Instruct"  # Replace with the actual model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize AirLLM for memory-efficient inference
air_llm = AirLLM(
    model_name=model_name,
    device="mps",
    memory_budget="30GB"
)

# Sample input text
# Sample input text
input_text = "Explain the concept of gravitational waves in simple terms."

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate predictions
outputs = air_llm.generate(
    inputs=inputs.input_ids,
    max_length=200,  # Set maximum token length for the response
    do_sample=True,  # Enable sampling for diverse outputs
    top_k=50,        # Top-k sampling
    top_p=0.95       # Nucleus sampling
)

# Decode and print the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Response:", response)