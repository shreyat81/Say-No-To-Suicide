import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = 'distilbert-base-uncased'
OUTPUT_DIR = './suicide-watch-model'
MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0

class SuicidalTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_clean_dataset(filepath):
    """Load and clean a dataset file with robust error handling."""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip')
                print(f"Successfully loaded {filepath} with {encoding} encoding")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
                
        if df is None or df.empty:
            print(f"Error: Could not read {filepath}")
            return None
            
        print(f"\nProcessing {filepath}:")
        print(f"Initial rows: {len(df)}")
        
        # Handle different column names
        text_cols = [col for col in df.columns if col.lower() in ['text', 'usertext', 'sentence', 'content']]
        label_cols = [col for col in df.columns if col.lower() in ['label', 'class', 'sentiment', 'is_suicidal']]
        
        if not text_cols or not label_cols:
            print(f"Error: Missing required columns in {filepath}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
            
        # Standardize column names
        df = df.rename(columns={
            text_cols[0]: 'text',
            label_cols[0]: 'label'
        })
        
        # Clean the data
        df = df[['text', 'label']].copy()
        df = df.dropna(subset=['text', 'label'])  # Remove rows with missing values
        df = df[df['text'].notna() & (df['text'].str.strip() != '')]  # Remove empty texts
        df = df.drop_duplicates(subset=['text'])  # Remove duplicate texts
        
        # Convert labels to integers (0 and 1)
        try:
            # First try to handle string labels
            if df['label'].dtype == 'object':
                # Convert to lowercase and strip whitespace
                df['label'] = df['label'].str.lower().str.strip()
                
                # Map common string labels to 0/1
                label_map = {
                    'suicide': 1, 'suicidal': 1, '1': 1, 'yes': 1, 'positive': 1, 'true': 1,
                    'non-suicide': 0, 'non-suicidal': 0, '0': 0, 'no': 0, 'negative': 0, 'false': 0,
                    'normal': 0, 'safe': 0, 'not suicidal': 0, 'not_suicidal': 0
                }
                
                # Map known labels, set others to NaN
                df['label'] = df['label'].map(label_map)
                
                # Drop rows with unmapped labels
                df = df.dropna(subset=['label'])
            
            # Convert to numeric (handles both string numbers and already converted)
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
            
            # Only keep valid labels (0 or 1)
            df = df[df['label'].isin([0, 1])]
            
            if df.empty:
                print(f"  Warning: No valid labels found in {filepath}")
                return None
                
            print(f"  Cleaned: {len(df)} samples (positive: {df['label'].sum()}, negative: {len(df) - df['label'].sum()})")
            return df
            
        except Exception as e:
            print(f"  Error processing labels in {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_model():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # List all available datasets
    dataset_files = [
        'suicidal_ideation_reddit_annotated.csv',
        'synthetic_dataset.csv',
        'synthetic_dataset_v2.csv'
    ]
    
    # Load and combine all valid datasets
    print("\nLoading and cleaning datasets...")
    dfs = []
    for file in dataset_files:
        if os.path.exists(file):
            df = load_and_clean_dataset(file)
            if df is not None and not df.empty:
                dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid datasets found to train on!")
    
    # Combine all datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Final cleanup and shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Check class distribution
    print("\nFinal dataset statistics:")
    print(f"Total samples: {len(combined_df)}")
    print("Class distribution:")
    print(combined_df['label'].value_counts())
    
    # Ensure we have both classes
    if len(combined_df['label'].unique()) < 2:
        raise ValueError("Dataset must contain both classes (0 and 1) for training!")
    
    # Split into training and validation
    train_df, val_df = train_test_split(
        combined_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=combined_df['label']
    )
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    # Create datasets
    train_dataset = SuicidalTextDataset(
        texts=train_df.text.tolist(),
        labels=train_df.label.tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = SuicidalTextDataset(
        texts=val_df.text.tolist(),
        labels=val_df.label.tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Calculate class weights for imbalanced dataset
    class_counts = train_df['label'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[cls] for cls in train_df['label']]
    
    # Create samplers
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': WEIGHT_DECAY,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
    
    # Learning rate scheduler
    total_steps = len(train_loader) // GRADIENT_ACCUMULATION_STEPS * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_val_accuracy = 0
    patience = 3
    patience_counter = 0
    global_step = 0
    
    for epoch in range(EPOCHS):
        print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')
        
        # Training
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader) - 1:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Calculate metrics
            with torch.no_grad():
                total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                _, predicted = torch.max(outputs.logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        # Calculate epoch metrics
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * total_correct / total_samples
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                val_loss += outputs.loss.item()
                _, preds = torch.max(outputs.logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Print epoch results
        print(f'\nEpoch {epoch + 1} Results:')
        print(f'Training Loss: {avg_train_loss:.4f} | Training Acc: {train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f} | Validation Acc: {val_accuracy:.2f}%')
        
        # Classification report
        print("\nValidation Report:")
        print(classification_report(all_labels, all_preds, target_names=['Not Suicidal', 'Suicidal']))
        
        # Early stopping with model checkpointing
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Create output directory if it doesn't exist
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
                
            # Save the best model
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f'\nNew best model saved with validation accuracy: {best_val_accuracy:.2f}%')
            
            # Save training arguments
            training_args = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_accuracy': best_val_accuracy,
                'total_steps': global_step
            }
            torch.save(training_args, os.path.join(OUTPUT_DIR, 'training_args.bin'))
        else:
            patience_counter += 1
            print(f'\nNo improvement in validation accuracy for {patience_counter}/{patience} epochs')
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    print('\nTraining complete!')

if __name__ == "__main__":
    train_model()
