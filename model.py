import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_model():
    api_key = "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"
    huggingface_hub.login(api_key)

    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer,model

# Load the LLaMA 3 model and tokenizer outside the Streamlit app
tokenizer, model = load_llama_model()