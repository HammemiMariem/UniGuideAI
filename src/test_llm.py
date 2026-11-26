
from llm import load_llm
import torch

tokenizer, model = load_llm()

print("\nMODEL INFORMATION")
print("-----------------")
print("Model:", model.config.name_or_path)
print("Architecture:", model.__class__.__name__)
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA available:", torch.cuda.is_available())

print("\nMODEL TEST")
print("----------")

messages = [
    {
        "role": "system",
        "content": "Answer clearly and accurately using the provided context."
    },
    {
        "role": "user",
        "content": (
            "Context: Erasmus Mundus supports joint Master programmes "
            "delivered by international consortia of universities.\n\n"
            "Question: What does Erasmus Mundus support?"
        )
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("Answer:", answer)

print(
    "\nGPU memory used:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)
