import streamlit as st
from model import tokenizer, model

# Set page title
st.title("LLaMA 3 Text Summarizer via Hugging Face")

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