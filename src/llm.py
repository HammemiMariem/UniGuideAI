
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_llm():

    print("Loading language model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model.eval()

    print("Language model loaded successfully!")
    print("Device: GPU")
    print("Quantization: 4-bit NF4")

    return tokenizer, model
