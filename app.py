from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

app = Flask(__name__)

print("Loading model...")
print("CUDA available:", torch.cuda.is_available())

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

print("Quantized model loaded successfully.")


def build_prompt(user_message: str) -> str:
    return f"""You are a helpful Anatomy and Physiology study assistant.
Answer briefly, clearly, and directly in 3-5 sentences.

User: {user_message}
Assistant:"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a message."}), 400

    prompt = build_prompt(user_message)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Assistant:" in full_text:
            reply = full_text.split("Assistant:")[-1].strip()
        else:
            reply = full_text.strip()

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)