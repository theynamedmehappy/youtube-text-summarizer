import streamlit as st
import transformers
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set page title
st.title("LLaMA 3 Text Summarizer via Hugging Face")

# Set the API key
api_key = "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"
huggingface_hub.login(api_key)

# Hugging Face Model and Tokenizer Loading
model_name = "meta-llama/Meta-Llama-3-8B"  # Change model as per need
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name,use_auth_token=True)


# Text input box
user_input = st.text_area("Enter text to summarize:")

# Function to summarize text
def summarize_text(input_text, max_length=200):
    inputs = tokenizer(input_text, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Summarize button
if st.button("Summarize"):
    if user_input:
        summary = summarize_text(user_input)
        st.subheader("Generated Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
