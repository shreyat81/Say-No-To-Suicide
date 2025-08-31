import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import argparse
import os

# --- Configuration ---
# Construct an absolute path to the model directory relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'suicide-watch-model')
MAX_LEN = 256

import json

def predict_suicidal_intent(text):
    """
    Predicts the suicidal intent of a given text and returns a score from 1 to 10
    along with alarming words based on attention scores.
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}") # Keep this quiet for API use

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, output_attentions=True)
    model.to(device)
    model.eval()

    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # --- Score Calculation ---
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    suicidal_prob = probabilities[0][1].item()
    score = 1 + (9 * (1 - (1 - suicidal_prob) ** 2))

    # --- Alarming Words Calculation (Attention-based) ---
    attentions = outputs.attentions  # (layers, batch_size, heads, seq_len, seq_len)
    # To get the attention weights, we average the attention from the [CLS] token 
    # across all attention heads and all layers.
    # attentions is a tuple of (batch_size, num_heads, seq_len, seq_len) tensors, one for each layer
    attention_heads = torch.stack(attentions).permute(1, 0, 2, 3, 4) # (batch_size, layers, heads, seq_len, seq_len)
    cls_attention = attention_heads[0, :, :, 0, :] # Attention from [CLS] token: (layers, heads, seq_len)
    cls_attentions = cls_attention.mean(dim=[0, 1]) # Average across layers and heads: (seq_len,)
    
    # Ignore attention for very short texts which might be false positives
    if len(text.split()) < 5 and score > 5:
        score = max(3.0, score - 2.0)  # Reduce score for very short texts

    # Normalize
    cls_attentions = cls_attentions / cls_attentions.sum()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Get words with attention above a certain threshold (e.g., 1.5x the mean)
    threshold = cls_attentions.mean() * 1.5
    alarming_words_indices = (cls_attentions > threshold).nonzero().squeeze().tolist()
    if not isinstance(alarming_words_indices, list):
        alarming_words_indices = [alarming_words_indices] # Handle single-item tensor case

    alarming_words = [tokens[i] for i in alarming_words_indices if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
    # Clean up subword tokens
    alarming_words = [word.replace('##', '') for word in alarming_words]

    return score, suicidal_prob, list(set(alarming_words)) # Return unique words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect suicidal ideation in text.')
    parser.add_argument('text', type=str, help='The text to analyze.')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format.')
    args = parser.parse_args()

    score, probability, alarming_words = predict_suicidal_intent(args.text)

    if args.json:
        # For API use, print a JSON object
        print(json.dumps({
            'score': score,
            'probability': probability,
            'alarmingWords': alarming_words
        }))
    else:
        # For command-line execution, print in a human-readable format
        print(f"\nInput Text: '{args.text}'")
        print(f"Suicidal Probability: {probability:.4f}")
        print(f"Suicidality Score (1-10): {score:.2f}")
        print(f"Alarming Words: {alarming_words}")
