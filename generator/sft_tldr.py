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
from peft import LoraConfig, get_peft_model, TaskType

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

        # 3. Configure LoRA adapters
        lora_config = LoraConfig(
            r=16, # Rank of the update matrices. Lower rank means fewer parameters to train.
            lora_alpha=32, # Alpha parameter for scaling.
            # Specify the modules to apply LoRA to. Common choices for transformers are 'q_proj' and 'v_proj'.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # 4. Create the PEFT model by wrapping the base model with LoRA config
        self.model = get_peft_model(base_model, lora_config)

        # Optional: Print the number of trainable parameters to see the difference
        self.model.print_trainable_parameters()
        self.model.train()

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
            # REVERSE: TL;DR is the prompt, post is the ground truth
            question = sample["completion"]
            ground_truth_answer = sample["prompt"]

            #no chat template is using a base model
            prompt_text = question
            prompt_text = "New post:\nTLDR:"+prompt_text+"\nSUBREDDIT:"

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=1,
                    do_sample=True,
                    top_p=1,
                    pad_token_id=self.tokenizer.pad_token_id
                    eos_token_id=self.tokenizer.eos_token_id
                )

            prompt_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0][prompt_length:]
            model_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f"--- SAMPLE {i+1}/{num_samples} (Example #{idx}) ---")
            print(f"QUESTION (TL;DR):\n{question}\n")
            print(f"GROUND TRUTH (Post):\n{ground_truth_answer}\n")
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
batch_size = 1
gradient_accumulation_steps = 4
block_size = 2048  # << increased to handle longer posts
num_epochs = 1
weight_decay = 0.1
max_lr = 3e-5
min_lr = 3e-6
warmup_ratio = 0.1
verbose = True
do_checkpoint = True
hub_model_id = "ultra-grok/model_tldrreverse"  # << changed

model_name = "NousResearch/Llama-2-7b-hf"
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


train_data = load_dataset("trl-lib/tldr", split="train")

input_ids_list = []
labels_list = []

# --- Data Packing Strategy ---
for item in tqdm(train_data):
    # REVERSE: TL;DR becomes the prompt, post becomes the answer
    question = item["completion"]
    answer = item["prompt"]
    prompt_str = "New post:\nTLDR:"+question+"\nSUBREDDIT:"
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_str = prompt_str + answer + tokenizer.eos_token
    full_ids = tokenizer.encode(full_str, add_special_tokens=False)
    this_labels = ([-100] * len(prompt_ids)) + full_ids[len(prompt_ids):]
    input_ids_list.extend(full_ids)
    labels_list.extend(this_labels)

long_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
long_labels = torch.tensor(labels_list, dtype=torch.long)

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

for epoch in range(num_epochs):
    lm.model.train()
    pbar = tqdm(range(num_steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}")
    for step in pbar:
        if verbose:
            if step%1500 == 300:
                lm.generate_samples(
                    dataset=train_data,
                    sample_indices=eval_sample_indices,
                    num_samples=num_eval_samples
            )
        max_start_idx = len(long_input_ids) - block_size - 1
        start_indices = [random.randint(0, max_start_idx) for _ in range(batch_size)]
        input_ids = torch.stack([long_input_ids[i:i+block_size] for i in start_indices])
        labels = torch.stack([long_labels[i:i+block_size] for i in start_indices])
        attention_mask = torch.ones_like(input_ids)

        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

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
            pbar.set_postfix({"custom_loss": f"{loss_val:.4f}", "lr": f"{lr:.2e}"})
            with open(log_file, "a") as f:
                f.write(f'Epoch {epoch+1}, Step {global_step}, Loss: {loss_val:.4f}, LR: {lr:.2e}\n')

            global_step += 1
            if do_checkpoint:
            # === Mid-training push to the Hub ===
                if global_step%1000 == 1:
                    rev = str(global_step//1000)
                    print(f"\nPushing adapter to Hub at mid-point (Step {global_step})...")
                    lm.model.push_to_hub(
                        hub_model_id, 
                        commit_message=f"mid-training push at step {global_step}",
                        revision="sft"+rev
                    )
                    lm.tokenizer.push_to_hub(
                        hub_model_id, 
                        commit_message=f"mid-training push at step {global_step}",
                        revision="sft"+rev
                    )
                    print(f"Successfully pushed revision 'sft{rev}' to the Hub.")
                    lm.model.train() # Ensure model is back in training mode
    
    if verbose:
        lm.generate_samples(
            dataset=train_data,
            sample_indices=eval_sample_indices,
            num_samples=num_eval_samples
        )

    # === Final push to the Hub ===
    if do_checkpoint:
        print("\nPushing final adapter to the Hub as revision '2sft'...")
        lm.model.push_to_hub(
            hub_model_id, 
            commit_message="End of training",
            revision="2sft"
        )
        lm.tokenizer.push_to_hub(
            hub_model_id, 
            commit_message="End of training",
            revision="2sft"
        )
        print("Final model (revision '2sft') successfully pushed to the Hub!")
print("Training finished!")
