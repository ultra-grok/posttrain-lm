import os
import math
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel


def get_lr(it, max_steps, warmup_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


class LanguageModel:
    def __init__(self, model_name, device, hub_model_id, adapter_revision="2sft"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model in bfloat16 + flash-attn2
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # attn_implementation="flash_attention_2"
        )

        # Load the PEFT adapter from hub
        self.model = PeftModel.from_pretrained(base_model, hub_model_id, revision=adapter_revision)

        # Set adapter weights to be trainable
        self.model.train()
        for n, p in self.model.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Attach a discriminator head
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_sizes", [])[0]
        self.model.discriminator_head = nn.Linear(hidden_size, 1, bias=True).to(self.device)
        for p in self.model.discriminator_head.parameters():
            p.requires_grad = True

    def configure_optimizer(self, weight_decay, max_lr):
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=max_lr)
        return optimizer


# ========================
# Setup
# ========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

max_grad_norm = 1.0
batch_size = 2
gradient_accumulation_steps = 2
block_size = 4096
num_epochs = 1
weight_decay = 0.1
max_lr = 3e-5
min_lr = 3e-6
warmup_ratio = 0.1

model_name = "NousResearch/Meta-Llama-3-8B"
hub_model_id = "ultra-grok/model_tldrreverse"
ADAPTER_REVISION = "2sft"

lm = LanguageModel(model_name, device, hub_model_id, adapter_revision=ADAPTER_REVISION)
lm.model.train()
tokenizer = lm.tokenizer

optimizer = lm.configure_optimizer(weight_decay, max_lr)
optimizer.zero_grad(set_to_none=True)

log_dir = "./disc_head_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

checkpoint_dir = os.path.join(log_dir, "latest_checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)

# ========================
# Load and split datasets
# ========================
HUB_DATASET_ID = "ultra-grok/tldr_sft_genreverse"
print(f"Loading dataset '{HUB_DATASET_ID}' (revision: {ADAPTER_REVISION})...")
ds_gen = load_dataset(HUB_DATASET_ID, revision=ADAPTER_REVISION)
ds_gen = ds_gen["validation"]

print("Loading dataset 'trl-lib/tldr' (validation split)...")
ds_true = load_dataset("trl-lib/tldr")["validation"]

# --- Create train and validation splits ---
N_TRUE_TRAIN = 6000
N_GEN_TRAIN = 9000

ds_true_train = ds_true.select(range(N_TRUE_TRAIN))
ds_true_val = ds_true.select(range(N_TRUE_TRAIN, len(ds_true)))
ds_gen_train = ds_gen.select(range(N_GEN_TRAIN))
ds_gen_val = ds_gen.select(range(N_GEN_TRAIN, len(ds_gen)))

print(f"Training set: {len(ds_true_train)} true examples, {len(ds_gen_train)} generated examples.")
print(f"Validation set: {len(ds_true_val)} true examples, {len(ds_gen_val)} generated examples.")


# ========================
# Build long sequence with labels for train and val
# ========================
def extract_completion_from_tldr_row(row):
    assert "completion" in row and row["completion"] is not None
    return row["completion"]

def process_sequences(ds_gen_split, ds_true_split):
    """Helper function to process datasets into sequences and labels."""
    sequences = []
    # Process generated data (label 0)
    for item in tqdm(ds_gen_split, desc="Encoding generated dataset"):
        question = item["prompt"]
        answer = item["completion"]
        prompt_ids = tokenizer.encode(question, add_special_tokens=False)
        full_str = question + answer + tokenizer.eos_token
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)
        gen_len = len(full_ids) - len(prompt_ids)
        bin_labels = [-100] * len(prompt_ids) + [0] * max(gen_len, 0)
        sequences.append((full_ids, bin_labels))

    # Process true data (label 1)
    for item in tqdm(ds_true_split, desc="Encoding true dataset"):
        answer = item["prompt"]
        question = "New post:\nTLDR:" + extract_completion_from_tldr_row(item) + "\nSUBREDDIT:"
        prompt_ids = tokenizer.encode(question, add_special_tokens=False)
        full_str = question + answer + tokenizer.eos_token
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)
        gen_len = len(full_ids) - len(prompt_ids)
        bin_labels = [-100] * len(prompt_ids) + [1] * max(gen_len, 0)
        sequences.append((full_ids, bin_labels))

    random.shuffle(sequences)
    return sequences

def create_long_tensors(sequences):
    """Helper function to concatenate sequences into long tensors."""
    input_ids_list = []
    bin_labels_list = []
    for full_ids, bin_labels in sequences:
        input_ids_list.extend(full_ids)
        bin_labels_list.extend(bin_labels)
    
    long_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    long_bin_labels = torch.tensor(bin_labels_list, dtype=torch.long)
    assert len(long_input_ids) == len(long_bin_labels)
    return long_input_ids, long_bin_labels

# Create training data
train_sequences = process_sequences(ds_gen_train, ds_true_train)
long_input_ids_train, long_bin_labels_train = create_long_tensors(train_sequences)

# Create validation data
val_sequences = process_sequences(ds_gen_val, ds_true_val)
long_input_ids_val, long_bin_labels_val = create_long_tensors(val_sequences)

# ========================
# Training
# ========================
tokens_per_batch = batch_size * block_size
num_steps_per_epoch = len(long_input_ids_train) // tokens_per_batch
total_steps = (num_steps_per_epoch // gradient_accumulation_steps) * num_epochs
warmup_steps = int(warmup_ratio * total_steps)
print(f"Starting training for {num_epochs} epoch(s). Total steps: {total_steps}")

global_step = 0
epsilon = 1e-8
bce_logits = nn.BCEWithLogitsLoss(reduction='none')

for epoch in range(num_epochs):
    lm.model.train()
    pbar = tqdm(range(num_steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}")
    for step in pbar:
        max_start_idx = len(long_input_ids_train) - block_size
        start_indices = [random.randint(0, max_start_idx) for _ in range(batch_size)]
        input_ids = torch.stack([long_input_ids_train[i:i+block_size] for i in start_indices])
        labels_bin = torch.stack([long_bin_labels_train[i:i+block_size] for i in start_indices])
        attention_mask = torch.ones_like(input_ids)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_bin = labels_bin.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            last_hidden = outputs.hidden_states[-1]
            disc_logits = lm.model.discriminator_head(last_hidden).squeeze(-1)

            valid_mask = (labels_bin != -100)
            if valid_mask.any():
                targets = labels_bin.float()
                per_token_loss = bce_logits(disc_logits, targets)
                masked_loss = (per_token_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(epsilon)
                loss = masked_loss / gradient_accumulation_steps
            else:
                loss = (disc_logits.sum() * 0.0)

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(lm.model.parameters(), max_grad_norm)
            lr = get_lr(global_step, total_steps, warmup_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item() * gradient_accumulation_steps
            pbar.set_postfix({
                "disc_loss": f"{loss_val:.4f}",
                "lr": f"{lr:.2e}"
            })
            with open(log_file, "a") as f:
                f.write(f'Epoch {epoch+1}, Step {global_step}, DiscLoss: {loss_val:.4f}, LR: {lr:.2e}\n')

            global_step += 1

    print(f"\nSaving checkpoint for epoch {epoch+1} to {checkpoint_dir}...")
    lm.model.save_pretrained(checkpoint_dir)
    lm.tokenizer.save_pretrained(checkpoint_dir)
    torch.save(lm.model.discriminator_head.state_dict(), os.path.join(checkpoint_dir, "discriminator_head.pt"))
    print(f"Checkpoint saved successfully.")

print("Training finished!")
# ========================
# Inference on Validation Set
# ========================
import json
from tqdm import tqdm

lm.model.eval()
torch.set_grad_enabled(False)

eos_id = tokenizer.eos_token_id

# 1) Truncate long stream to a multiple of block_size (never pad)
N_total = len(long_input_ids_val)
N_trunc = (N_total // block_size) * block_size
if N_trunc == 0:
    raise RuntimeError("Truncated stream length is 0 (block_size too large)")

stream = long_input_ids_val[:N_trunc]  # CPU tensor

# 2) Find first and last eos in truncated stream and define the valid inclusive interval
eos_positions = (stream == eos_id).nonzero(as_tuple=True)[0]
if eos_positions.numel() == 0:
    print("No EOS token found in truncated stream — nothing to save.")
else:
    first_eos = int(eos_positions[0].item())
    last_eos = int(eos_positions[-1].item())
    global_start = first_eos + 1      # discard anything before first eos
    global_end = last_eos             # discard anything after last eos (inclusive)

    # 3) Prepare arrays to hold per-token logits/probs (overwrite on each write)
    logits_all = torch.full((N_trunc,), float("nan"), dtype=torch.float32)
    probs_all  = torch.full((N_trunc,), float("nan"), dtype=torch.float32)

    # 4) Make overlapping start positions and batch them (no padding; drop incomplete final batch)
    stride = max(1, block_size // 2)  # overlap; reasonable default
    starts = list(range(0, N_trunc - block_size + 1, stride))
    # group into batches of size `batch_size`
    batched_starts = [starts[i:i + batch_size] for i in range(0, len(starts), batch_size)]
    # drop last batch if it's incomplete (keeps logic simple and avoids padding)
    if len(batched_starts) and len(batched_starts[-1]) < batch_size:
        batched_starts = batched_starts[:-1]

    print(f"Stream length (truncated): {N_trunc}. windows: {len(starts)}, batches: {len(batched_starts)}")

    # 5) Run model over each batch of windows and overwrite the corresponding slice in logits_all/probs_all
    for batch_starts in tqdm(batched_starts, desc="Stream inference (overlap)"):
        # stack blocks -> shape (B, block_size)
        batch_inputs = torch.stack([stream[s:s + block_size] for s in batch_starts]).to(device)
        attention_mask = torch.ones_like(batch_inputs, dtype=torch.long).to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = lm.model(
                        input_ids=batch_inputs,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False
                    )
            else:
                outputs = lm.model(
                    input_ids=batch_inputs,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
            last_hidden = outputs.hidden_states[-1]                # (B, block_size, H)
            disc_logits = lm.model.discriminator_head(last_hidden.to(lm.model.discriminator_head.weight.dtype)).squeeze(-1)
            disc_probs = torch.sigmoid(disc_logits)                # (B, block_size)

        disc_logits = disc_logits.cpu().float()
        disc_probs  = disc_probs.cpu().float()

        # overwrite results for each window (later windows replace earlier ones)
        for i, s in enumerate(batch_starts):
            logits_all[s:s + block_size] = disc_logits[i]
            probs_all[s:s + block_size]  = disc_probs[i]

    # 6) Iterate original `val_sequences` and save only complete examples
    out_dir = "./disc_inference_results"
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "inference_results.jsonl")

    pos = 0
    written = 0
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for full_ids, bin_labels in tqdm(val_sequences, desc="Extract & save examples"):
            L = len(full_ids)
            # skip sequences that extend beyond the truncated stream (we never padded)
            if pos + L > N_trunc:
                pos += L
                continue

            # enforce the "throw away anything before first eos and after last eos"
            if pos < global_start or (pos + L - 1) > global_end:
                pos += L
                continue

            labels_slice = long_bin_labels_val[pos:pos + L].tolist()
            # must have at least one completion token (label != -100)
            first_completion_idx = next((i for i, v in enumerate(labels_slice) if v != -100), None)
            if first_completion_idx is None:
                pos += L
                continue

            ids_slice = stream[pos:pos + L].tolist()
            # require final token be EOS (complete example)
            if ids_slice[-1] != eos_id:
                pos += L
                continue

            # gather corresponding logits/probs (these have been overwritten by inference)
            logits_slice = logits_all[pos:pos + L].tolist()
            probs_slice  = probs_all[pos:pos + L].tolist()

            prompt_ids = ids_slice[:first_completion_idx]
            completion_ids = ids_slice[first_completion_idx:]

            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)

            entry = {
                "prompt": prompt_text,
                "completion": completion_text,
                "token_ids": ids_slice,
                "token_logits": logits_slice,
                "token_probs": probs_slice,
                "labels": labels_slice
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            pos += L

    print(f"Done — wrote {written} complete examples to: {out_jsonl}")

# ========================
# Push results to Hub
# ========================
from datasets import load_dataset
from huggingface_hub import notebook_login, HfApi

# --- Configuration ---
# This is the dataset repo on the Hub you want to push to
HUB_DATASET_ID = "ultra-grok/tldr_sft_genreverse"
# This is the local file generated by the previous inference step
LOCAL_JSONL_PATH = out_jsonl
# This is the name for your new branch/revision on the Hub
NEW_REVISION_NAME = "rm_split"

print("\nStarting dataset push to the Hugging Face Hub...")

# --- Step 1: Authenticate ---
# Make sure you are logged in. You can do this in your terminal beforehand:
# huggingface-cli login
# Or, if in a notebook, uncomment the line below:
# notebook_login()

# --- Step 2: Load the local JSONL file as a Hugging Face Dataset object ---
print(f"Loading local file: {LOCAL_JSONL_PATH}")
# The 'split' is automatically named 'train' by default when loading a single file
inference_dataset = load_dataset('json', data_files=LOCAL_JSONL_PATH)

# --- Step 3: Push the dataset to the Hub under the specified revision ---
print(f"Pushing dataset to '{HUB_DATASET_ID}' on revision '{NEW_REVISION_NAME}'...")
inference_dataset.push_to_hub(
    repo_id=HUB_DATASET_ID,
    revision=NEW_REVISION_NAME,
    commit_message=f"Add inference results for reward model training"
)

print("✅ Push complete!")
print(f"You can find your new dataset revision at: https://huggingface.co/datasets/{HUB_DATASET_ID}/tree/{NEW_REVISION_NAME}")
