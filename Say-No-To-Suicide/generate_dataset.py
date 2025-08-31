import requests
import json
import pandas as pd
import time
import random
from tqdm import tqdm

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"  # Using one of the models you have installed
NUM_SAMPLES_PER_CLASS = 200  # Further increased sample count for better coverage
OUTPUT_FILE = 'synthetic_dataset_v2.csv' # New output file


def generate_text(prompt, model_name, temperature=0.7):
    """Generates text using a local Ollama model."""
    # System prompt to guide the model's output
    full_prompt = f"You are a helpful assistant that generates text samples for a machine learning dataset. Please only output the text sample itself, with no extra commentary or quotation marks. Here is the request: {prompt}"

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()['response'].strip()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred connecting to Ollama: {e}")
        print("Please ensure the Ollama server is running and accessible.")
        # Simple backoff
        time.sleep(5)
        return None

def main():
    """Main function to generate and save the dataset."""

    # More specific and varied suicidal ideation prompts
    suicidal_prompts = [
        "I can't take this pain anymore. Every day is a struggle and I don't see the point in continuing.",
        "I'm such a burden to everyone around me. They'd be better off without me.",
        "I've been thinking about ending it all. I've even started making plans.",
        "Goodbye world. This is my last message. Please tell my family I'm sorry.",
        "The pain is too much. I just want it to stop. Permanently.",
        "I feel so alone and hopeless. No one would miss me if I was gone.",
        "I've written my suicide note. I just need to find the courage to go through with it.",
        "Life has no meaning anymore. I'm ready to end it all.",
        "I can't handle this pain any longer. I'm ready to say goodbye.",
        "I've been researching painless ways to die. I think I've found one that will work."
    ]

    # More diverse positive/neutral examples, including hard negatives
    non_suicidal_prompts = [
        # Standard positive/neutral examples
        "I'm feeling great today! Just got a promotion at work.",
        "Can't wait for the weekend. Planning to go hiking with friends.",
        "Just finished reading an amazing book. Highly recommend it!",
        
        # Hard negatives - contain concerning words but in safe contexts
        "I feel like I'm dying of laughter after watching that comedy show!",
        "This workout is killing me, but I love the burn!",
        "I could just die for a piece of that chocolate cake right now.",
        "I'm so tired I could sleep for a thousand years.",
        "I feel like I'm drowning in work, but I'll get through it.",
        "That horror movie was to die for! The special effects were amazing.",
        "I'm dead serious about acing this exam - been studying all week!",
        "This heat is killing me. Can't wait for winter!",
        "I'm so hungry I could eat a horse right now.",
        "That rollercoaster was a near-death experience, but so much fun!"
    ]

    print(f"Generating {NUM_SAMPLES_PER_CLASS * 2} samples using {MODEL_NAME}...")
    data = []

    # Generate Suicidal Samples
    print("\nGenerating 'suicidal' samples...")
    for _ in tqdm(range(NUM_SAMPLES_PER_CLASS), desc="Suicidal"):
        prompt = random.choice(suicidal_prompts)
        text = generate_text(prompt, MODEL_NAME, temperature=0.9)
        if text:
            data.append({'text': text, 'class': 'suicide'})

    # Generate Non-Suicidal Samples
    print("\nGenerating 'non-suicidal' samples...")
    for _ in tqdm(range(NUM_SAMPLES_PER_CLASS), desc="Non-Suicidal"):
        prompt = random.choice(non_suicidal_prompts)
        text = generate_text(prompt, MODEL_NAME, temperature=0.7)
        if text:
            data.append({'text': text, 'class': 'non-suicide'})

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSuccessfully generated {len(df)} samples and saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
