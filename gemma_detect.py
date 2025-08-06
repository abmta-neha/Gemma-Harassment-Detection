from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_gemma_prediction(voice_text, heart_rate, stress, temp, eda):
    prompt = f"""The following are physiological indicators and voice input:
    Voice: "{voice_text}"
    Heart rate: {heart_rate}
    Stress level: {stress}
    Skin temperature: {temp}
    EDA: {eda}
    Based on this, is the user likely in danger of harassment? Answer 'Yes' or 'No' and explain briefly."""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
