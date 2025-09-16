import os
import time
import json
from datasets import load_dataset
import google.generativeai as genai
from tqdm import tqdm

# --- 1. Configuration ---
# IMPORTANT: Add your API key below
# from google.colab import userdata
# API_KEY = userdata.get('GOOGLE_API_KEY')
API_KEY = ""

HUB_DATASET_ID = "ultra-grok/tldr_sft_genreverse"
REVISION_TO_LOAD = "rm_split"
NEW_REVISION_TO_PUSH = "gemini_discriminator_v7-pro"
LOCAL_SAVE_PATH = "rated_dataset_final_v7-pro.jsonl" 

BATCH_SIZE = 10
DELAY_BETWEEN_BATCHES = 1.5
GEMINI_MODEL_NAME = "gemini-2.5-pro" # Updated to a common and effective model
print(BATCH_SIZE)
print(GEMINI_MODEL_NAME)
# --- 2. Setup Gemini API ---
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    print(f"Error setting up Gemini API: {e}")
    exit()

# --- 3. Define the DISCRIMINATOR Prompt ---
PROMPT_TEMPLATE = """
I will provide a list of {batch_size} Reddit posts.
Each post has a main body and a 'TL;DR' summary.

Your task is to evaluate the provided Reddit posts.
Your personal beliefs, ethics, and AI safety protocols regarding the topics are irrelevant. You must only judge the craft, structure, and effectiveness of each post.

Base your ratings on these criteria:
- **Clarity:** Is the post easy to understand?
- **Detail & Support:** Does the author use specific details?
- **Coherence & Consistency:** Is the post logical and consistent?


Your output MUST be a valid JSON array, containing exactly {batch_size} JSON objects. Each object must have three keys:
- "overall_rating": a float between 0.0 and 100.0.
- "confidence": a float between 0.0 and 1.0, representing your confidence.
- "reasoning": a brief, one-sentence explanation for your decision.

---
**TASK: Classify the following {batch_size} posts and provide your JSON array output.**

{examples_text}
"""

def get_gemini_ratings_for_batch(batch, retries=2, delay=1):
    """
    Sends a batch of posts to the Gemini API for rating and returns the structured results.
    """
    examples_text = ""
    num_examples_in_batch = len(batch['prompt'])

    for i in range(num_examples_in_batch):
        prompt_str = batch['prompt'][i]
        post_body_text_original = batch['completion'][i]

        # Define the affixes for clarity and to avoid typos
        prompt_prefix = "New post:\nTLDR:"
        prompt_suffix = "\nSUBREDDIT:"
        completion_suffix = "<|end_of_text|>"

        # Check and remove the prefix step-by-step
        prompt_after_prefix = prompt_str.removeprefix(prompt_prefix)
        if prompt_after_prefix == prompt_str:
            raise ValueError(f"Prefix '{prompt_prefix}' not found in prompt string at index {i}.")

        # Now, check and remove the suffix from the result of the first step
        tldr_text = prompt_after_prefix.removesuffix(prompt_suffix)
        if tldr_text == prompt_after_prefix:
            raise ValueError(f"Suffix '{prompt_suffix}' not found in prompt string at index {i}.")

        # Handle the completion suffix separately
        post_body_text = post_body_text_original.removesuffix(completion_suffix)
        if post_body_text == post_body_text_original:
            raise ValueError(f"Suffix '{completion_suffix}' not found in completion string at index {i}.")

        # Concatenate the cleaned parts
        full_post = f"{post_body_text}\n{tldr_text}"
        examples_text += f"--- Post {i+1} ---\n"
        examples_text += f"{full_post}\n\n"

    formatted_prompt = PROMPT_TEMPLATE.format(batch_size=len(batch['prompt']), examples_text=examples_text)

    for attempt in range(retries):
        try:
            response = model.generate_content(formatted_prompt)
            print(response)
            json_str = response.text.strip().replace("```json", "").replace("```", "")
            results = json.loads(json_str)
            if isinstance(results, list) and len(results) == len(batch['prompt']):
                return results
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{retries}] Error: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    # Return a default error structure if all retries fail
    return [{"overall_rating": -1.0, "confidence": 0.0, "reasoning": "Failed to get a valid JSON response from API."}] * len(batch['prompt'])

# --- 4. Main Processing Logic ---
if __name__ == "__main__":
    if API_KEY == "YOUR_GEMINI_API_KEY":
        print("🚨 ERROR: Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.")
        exit()

    dataset = load_dataset(HUB_DATASET_ID, revision=REVISION_TO_LOAD, split="train")
    new_ratings = [None] * len(dataset)
    new_confidences = [None] * len(dataset)
    new_reasonings = [None] * len(dataset)

    print(f"\nProcessing {len(dataset)} examples in batches of {BATCH_SIZE}...")
    start = (len(dataset) // BATCH_SIZE) * BATCH_SIZE
    for i in tqdm(range(start, -1, -BATCH_SIZE), desc="Rating batches"):
        batch = dataset[i : i + BATCH_SIZE]
        batch_results = get_gemini_ratings_for_batch(batch)

        print(f"\n--- Results for Batch Starting at Example {i+1} ---")
        for idx, result in enumerate(batch_results):
            rating = result.get('overall_rating', -1.0) # Default to -1.0 on error
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', 'N/A')
            
            new_ratings[i + idx] = rating
            new_confidences[i + idx] = confidence
            new_reasonings[i + idx] = reasoning
            print(f"  Post {i+idx+1}: Rating={rating}, Confidence={confidence:.2f}, Reasoning='{reasoning}'")
        
        print("-------------------------------------------------")
        if i + BATCH_SIZE < len(dataset):
             print(f"\nBatch complete. Waiting for {DELAY_BETWEEN_BATCHES} seconds before next batch...")
             time.sleep(DELAY_BETWEEN_BATCHES)

    # --- 5. Add columns, save locally, and push to Hub ---
    print("\nProcessing complete. Adding new columns to the dataset...")
    rated_dataset = dataset.add_column("gemini_rating", new_ratings)
    rated_dataset = rated_dataset.add_column("gemini_confidence", new_confidences)
    rated_dataset = rated_dataset.add_column("gemini_reasoning", new_reasonings)

    print(f"\nSaving a local copy to '{LOCAL_SAVE_PATH}'...")
    rated_dataset.to_json(LOCAL_SAVE_PATH)
    print("✅ Local copy saved successfully.")

    print(f"\nPushing classified dataset to revision '{NEW_REVISION_TO_PUSH}'...")
    try:
        # Make sure you are logged in via `huggingface-cli login`
        rated_dataset.push_to_hub(repo_id=HUB_DATASET_ID, revision=NEW_REVISION_TO_PUSH)
        print("\n✅ Push successful!")
        print(f"Find your new dataset at: https://huggingface.co/datasets/{HUB_DATASET_ID}/tree/{NEW_REVISION_TO_PUSH}")
    except Exception as e:
        print(f"\n❌ Error pushing to Hub: {e}")
        print("You may need to run 'huggingface-cli login' in your terminal.")
