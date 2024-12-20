import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import tensorflow
import torch

api_key = "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"
login(api_key)

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_auth_token=api_key)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=api_key)

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
)

user_feed = input()
messages = [
    {"role": "system", "content": "Assume you are a doctor and have extensive knowledge of disease and its remedies so you will be\
                                   given a text with some symptoms and you have analyze that symptoms and predcit the what is diseases\
                                   user might have and recoomed a short remedies for it in keywords nots a sentence"},
    {"role": "user", "content": user_feed}
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

for entry in outputs:
    if 'generated_text' in entry:
        for text in entry['generated_text']:
            if text['role'] == 'assistant':
                print(text['content'])