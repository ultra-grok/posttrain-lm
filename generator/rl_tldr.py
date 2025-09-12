import os
import math
import random
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, load_dataset
#peft
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def get_lr(it, max_steps, warmup_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

class LanguageModel:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        #peft
        # 1. Configure quantization to load the model in 4-bit for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. Load the base model with the quantization config
        # device_map="auto" will handle placing the model on the correct device (e.g., GPU)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

        # Load the PEFT adapter from hub
        self.model = PeftModel.from_pretrained(base_model, hub_model_id, revision="2sft")

        # Optional: Print the number of trainable parameters to see the difference
        self.model.train()
        self.model.print_trainable_parameters()
        
        for n, p in self.model.named_parameters():
            p.requires_grad = ("lora_" in n) or ("lora_A" in n) or ("lora_B" in n)

        self.model.train()

        # LOAD THE REFERENCE MODEL
        print("Loading frozen reference model...")
        ref_base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
        self.ref_model = PeftModel.from_pretrained(ref_base_model, hub_model_id, revision="2sft")
        # Freeze the reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

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

    def generate_samples(self, dataset, sample_indices, num_samples=16):
        print(f"\nGenerating {num_samples} samples for evaluation...\n")
        self.model.eval()

        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            question = sample["prompt"]
            ground_truth_answer = sample["completion"]

            #no chat template for base model
            prompt_text =question

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,  # << increased from 300
                    temperature=1,
                    do_sample=True,
                    top_p=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            prompt_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0][prompt_length:]
            model_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f"--- SAMPLE {i+1}/{num_samples} (Example #{idx}) ---")
            print(f"QUESTION:\n{question}\n")
            print(f"GROUND TRUTH:\n{ground_truth_answer}\n")
            print(f"MODEL OUTPUT:\n{model_answer.strip()}\n")
            print("="*50 + "\n")

        self.model.train()
        print("Generation complete! Resuming training...")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

max_grad_norm = 1.0
batch_size = 2
gradient_accumulation_steps = 2
block_size = 2048
num_epochs = 1
weight_decay = 0.1
max_lr = 3e-5
min_lr = 3e-6
warmup_ratio = 0.1
ppo_clip_epsilon = 0.3

model_name = "NousResearch/Llama-3.2-1B"
hub_model_id = "ultra-grok/model_tldrreverse"  # << changed
lm = LanguageModel(model_name, device)
lm.model.train()
tokenizer = lm.tokenizer

optimizer = lm.configure_optimizer(weight_decay, max_lr)
optimizer.zero_grad(set_to_none=True)

log_dir = "./sft_model_logs_improved"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f: pass

checkpoint_dir = os.path.join(log_dir, "latest_checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)


HUB_DATASET_ID = "ultra-grok/tldr_sft_genreverse"  # << changed
ADAPTER_REVISION = "2sft"

# --- Load the specific revision of the dataset from the Hub ---
print(f"Loading dataset '{HUB_DATASET_ID}' (revision: {ADAPTER_REVISION})...")
train_data = load_dataset(
    HUB_DATASET_ID,
    revision=ADAPTER_REVISION
)
train_data = train_data["validation"]

input_ids_list = []
labels_list = []
advantages_list = []

for item in tqdm(train_data):
    question = item["prompt"]
    answer = item["completion"]
    # creating a reward that peaks at 1 for average-length summaries.
    score = 1 - abs(item["score"])
    prompt_str = question
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_str = prompt_str + answer + tokenizer.eos_token
    full_ids = tokenizer.encode(full_str, add_special_tokens=False)

    generation_len = len(full_ids) - len(prompt_ids)
    # Avoid division by zero for empty generations
    token_advantage = score / generation_len if generation_len > 0 else 0.0

    labels_this = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    advantages_this = [0.0] * len(prompt_ids) + [token_advantage] * generation_len

    input_ids_list.extend(full_ids)
    labels_list.extend(labels_this)
    advantages_list.extend(advantages_this)

long_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
long_labels = torch.tensor(labels_list, dtype=torch.long)
long_advantages = torch.tensor(advantages_list, dtype=torch.float)

num_eval_samples = 2
total_examples = len(train_data)
eval_sample_indices = random.sample(range(total_examples), num_eval_samples)


# Calculate steps for one full pass over the packed dataset
tokens_per_batch = batch_size * block_size
num_steps_per_epoch = len(long_input_ids) // tokens_per_batch

total_steps = (num_steps_per_epoch // gradient_accumulation_steps) * num_epochs
warmup_steps = int(warmup_ratio * total_steps)
print(f"Starting training for {num_epochs} epoch(s). Total steps: {total_steps}")

global_step = 0
epsilon = 1e-8

lm.generate_samples(
    dataset=train_data,
    sample_indices=eval_sample_indices,
    num_samples=num_eval_samples
)
for epoch in range(num_epochs):
    lm.model.train()
    pbar = tqdm(range(num_steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}")
    for step in pbar:
        max_start_idx = len(long_input_ids) - block_size
        start_indices = [random.randint(0, max_start_idx) for _ in range(batch_size)]
        input_ids = torch.stack([long_input_ids[i:i+block_size] for i in start_indices])
        labels = torch.stack([long_labels[i:i+block_size] for i in start_indices])
        advantages = torch.stack([long_advantages[i:i+block_size] for i in start_indices])
        attention_mask = torch.ones_like(input_ids)

        input_ids, attention_mask, labels, advantages = input_ids.to(device), attention_mask.to(device), labels.to(device), advantages.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy_outputs = lm.model(input_ids=input_ids, attention_mask=attention_mask)
            policy_logits = policy_outputs.logits

        with torch.no_grad():
            ref_outputs = lm.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits

        policy_logits_shifted = policy_logits[:, :-1, :].contiguous()
        ref_logits_shifted = ref_logits[:, :-1, :].contiguous()
        labels_shifted = labels[:, 1:].contiguous()

        logprobs_policy = -F.cross_entropy(
            policy_logits_shifted.view(-1, policy_logits_shifted.shape[-1]),
            labels_shifted.view(-1),
            reduction='none',
            ignore_index=-100
        ).view(policy_logits.shape[0], -1)

        logprobs_ref = -F.cross_entropy(
            ref_logits_shifted.view(-1, ref_logits_shifted.shape[-1]),
            labels_shifted.view(-1),
            reduction='none',
            ignore_index=-100
        ).view(ref_logits.shape[0], -1)
        
        advantages_shifted = advantages[:, 1:].to(logprobs_policy.dtype)

        ratio = torch.exp(logprobs_policy - logprobs_ref)
        
        valid_mask = (labels_shifted != -100).float()

        adv_count = valid_mask.sum()
        if adv_count > 1:
            masked_advantages = advantages_shifted * valid_mask
            adv_mean = masked_advantages.sum() / adv_count
            adv_std = ((masked_advantages - adv_mean).pow(2) * valid_mask).sum() / adv_count
            advantages_shifted = (masked_advantages - adv_mean) / (adv_std.sqrt() + epsilon)

        advantages_shifted = torch.clamp(advantages_shifted, -3, 3)
        
        surr1 = ratio * advantages_shifted
        surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages_shifted

        ppo_loss = -torch.min(surr1, surr2)
        masked_ppo_loss = ppo_loss * valid_mask
        mean_ppo_loss = masked_ppo_loss.sum() / valid_mask.sum().clamp_min(epsilon)
        
        kl_beta = 0.01
        
        log_r = logprobs_ref - logprobs_policy
        
        r = torch.exp(log_r)
        
        kl_div = r - 1 - log_r
        
        masked_kl_div = kl_div * valid_mask
        mean_kl_div = masked_kl_div.sum() / valid_mask.sum().clamp_min(epsilon)
        
        loss = mean_ppo_loss + kl_beta * mean_kl_div
        
        loss = loss / gradient_accumulation_steps
        
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(lm.model.parameters(), max_grad_norm)
            lr = get_lr(global_step, total_steps, warmup_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item() * gradient_accumulation_steps
            kl_val = mean_kl_div.item()

            pbar.set_postfix({
                "custom_loss": f"{loss_val:.4f}", 
                "kl_div": f"{kl_val:.4f}",
                "lr": f"{lr:.2e}"
            })
            with open(log_file, "a") as f:
                f.write(f'Epoch {epoch+1}, Step {global_step}, Loss: {loss_val:.4f}, KL: {kl_val:.4f}, LR: {lr:.2e}\n')

            global_step += 1

    print(f"\nSaving checkpoint for epoch {epoch+1} to {checkpoint_dir}...")
    lm.model.save_pretrained(checkpoint_dir)
    lm.tokenizer.save_pretrained(checkpoint_dir)
    print(f"Checkpoint saved successfully.")

    lm.generate_samples(
        dataset=train_data,
        sample_indices=eval_sample_indices,
        num_samples=num_eval_samples
    )

print("\nPushing final adapter to the Hub as revision '1rl'...")
lm.model.push_to_hub(
    hub_model_id, 
    commit_message="End of training",
    revision="1rl"
)
lm.tokenizer.push_to_hub(
    hub_model_id, 
    commit_message="End of training",
    revision="1rl"
)
print("Final model (revision '1rl') successfully pushed to the Hub!")

print("Training finished!")
