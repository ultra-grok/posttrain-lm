import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Define configurations (adapt as needed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "NousResearch/Llama-3.2-1B"
CHECKPOINT_DIR = "ultra-grok/model_tldrreverse"  # << changed
DATASET_ID = "trl-lib/tldr"
SPLIT = "validation"  
NUM_EXAMPLES = 10000  # Adjust as needed
SEED = 42
MAX_NEW_TOKENS = 1024  # << increased for long posts
BATCH_SIZE = 8  # << reduced to fit memory with long generations
HUB_DATASET_ID = "ultra-grok/tldr_sft_genreverse"  # << changed
ADAPTER_REVISION = "2sft"  # Or specific revision
TEMPERATURE = 1
TOP_P = 1.0
hub_model_id = "ultra-grok/model_tldrreverse"  # << changed
verbose = False
# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, revision = ADAPTER_REVISION)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model with quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)

# Load PEFT model from checkpoint
model = PeftModel.from_pretrained(
    base_model,
    hub_model_id,
    revision=ADAPTER_REVISION,
    device_map="auto"
)
model.eval()

# Load and sample the dataset
print(f"Loading dataset '{DATASET_ID}'...")
dataset = load_dataset(DATASET_ID, split=SPLIT)

print(f"Shuffling and selecting {NUM_EXAMPLES} examples with seed {SEED}...")
shuffled_dataset = dataset.shuffle(seed=SEED)
N = len(shuffled_dataset)
# Sample with replacement to generate more examples than available in the source data
indices = np.random.choice(N, size=NUM_EXAMPLES, replace=True)
samples = shuffled_dataset.select(indices.tolist())

# Prepare prompts — reverse task: use TL;DR (completion) as the prompt
prompts = ["New post:\nTLDR:"+sample["completion"]+"\nSUBREDDIT:" for sample in samples]  # << swapped

# Generate completions in batches
results_data = []
raw_scores = []

print(f"Generating completions for {NUM_EXAMPLES} samples in batches...")
for start_idx in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch_prompts = prompts[start_idx:start_idx + BATCH_SIZE]

    # Tokenize with padding
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding_side='left', padding=True, truncation=False)
    inputs = inputs.to(DEVICE)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode all outputs at once
    decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Process each output individually to correctly strip the prompt
    for i, full_text in enumerate(decoded_outputs):
        prompt_text = batch_prompts[i]

        # Robustly remove the prompt from the beginning of the decoded text
        if full_text.startswith(prompt_text):
            completion = full_text[len(prompt_text):].strip()
        else:
            # Fallback if the model doesn't perfectly repeat the prompt
            completion = full_text.strip()

        # Score is based on the length of the completion, not the total length
        completion_len = len(completion)
        raw_scores.append(completion_len)

        results_data.append({
            "prompt": prompt_text,
            "completion": completion,
            "score": -completion_len  # Score is negative length of the summary
        })
        if verbose:
            print({
            "prompt": prompt_text,
            "completion": completion,
            "score": -completion_len  # Score is negative length of the summary
        })

# Standardize scores
raw_scores = np.array(raw_scores)
print(f"Mean completion length: {raw_scores.mean():.2f}")
print(f"Variance of completion length: {raw_scores.var():.2f}")

# Z-score normalization on -raw_scores
scores = -raw_scores
scores_std = (scores - scores.mean()) / (scores.std() + 1e-8)

for i in range(len(results_data)):
    results_data[i]["score"] = float(scores_std[i])

# Create dataset and push
final_dataset = DatasetDict({
    "validation": Dataset.from_list(results_data)
})

print(f"Pushing dataset to '{HUB_DATASET_ID}'...")
final_dataset.push_to_hub(
    repo_id=HUB_DATASET_ID,
    commit_message=f"Add standardized inference results for model revision {ADAPTER_REVISION}",
    revision=ADAPTER_REVISION,
    private=False
)
print("✅ Push complete!")
