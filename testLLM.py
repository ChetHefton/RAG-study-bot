from transformers import pipeline
import torch

print("CUDA available:", torch.cuda.is_available())

generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain anatomy and physiology in 3 sentences."

output = generator(prompt, max_new_tokens=150)

print(output[0]["generated_text"])