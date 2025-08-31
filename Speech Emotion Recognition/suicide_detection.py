import os
import warnings
warnings.filterwarnings("ignore")

# Try to import dependencies with fallback
try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import torch.nn.functional as F
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Suicide detection dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

class SuicideDetector:
    def __init__(self, model_path=None):
        """
        Initialize the suicide detection model
        """
        if model_path is None:
            # Default path relative to current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'Say-No-To-Suicide', 'suicide-watch-model')
        
        self.model_path = model_path
        self.max_len = 256
        
        # Check if dependencies are available
        if not DEPENDENCIES_AVAILABLE:
            self.model_loaded = False
            self.device = None
            self.model = None
            self.tokenizer = None
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer as None - will load on first use
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
    
    def _load_model(self):
        """
        Lazy loading of the model to avoid loading it if not needed
        """
        if not DEPENDENCIES_AVAILABLE:
            return False
            
        if not self.model_loaded:
            try:
                print(f"Loading suicide detection model from {self.model_path}")
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.model_path, 
                    output_attentions=True
                )
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                print("Suicide detection model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading suicide detection model: {e}")
                self.model_loaded = False
                return False
        return True
    
    def predict_suicidal_intent(self, text):
        """
        Predicts the suicidal intent of a given text and returns a score from 1 to 10
        along with alarming words based on attention scores.
        
        Returns:
            dict: {
                'score': float (1-10),
                'probability': float (0-1),
                'alarming_words': list,
                'risk_level': str ('low', 'medium', 'high', 'critical')
            }
        """
        if not text or not text.strip():
            return {
                'score': 1.0,
                'probability': 0.0,
                'alarming_words': [],
                'risk_level': 'low'
            }
        
        # Check if dependencies are available
        if not DEPENDENCIES_AVAILABLE:
            return {
                'score': 1.0,
                'probability': 0.0,
                'alarming_words': [],
                'risk_level': 'low',
                'error': 'Dependencies not available'
            }
        
        # Load model if not already loaded
        if not self.model_loaded:
            success = self._load_model()
            if not success:
                return {
                    'score': 1.0,
                    'probability': 0.0,
                    'alarming_words': [],
                    'risk_level': 'low',
                    'error': 'Model not available'
                }
        
        try:
            # Tokenize the input text
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

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Score calculation
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            suicidal_prob = probabilities[0][1].item()
            score = 1 + (9 * (1 - (1 - suicidal_prob) ** 2))

            # Alarming words calculation (attention-based)
            attentions = outputs.attentions
            attention_heads = torch.stack(attentions).permute(1, 0, 2, 3, 4)
            cls_attention = attention_heads[0, :, :, 0, :]
            cls_attentions = cls_attention.mean(dim=[0, 1])
            
            # Adjust score for very short texts
            if len(text.split()) < 5 and score > 5:
                score = max(3.0, score - 2.0)

            # Normalize attention
            cls_attentions = cls_attentions / cls_attentions.sum()

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Get alarming words
            threshold = cls_attentions.mean() * 1.5
            alarming_words_indices = (cls_attentions > threshold).nonzero().squeeze().tolist()
            if not isinstance(alarming_words_indices, list):
                alarming_words_indices = [alarming_words_indices]

            alarming_words = [tokens[i] for i in alarming_words_indices 
                            if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
            alarming_words = [word.replace('##', '') for word in alarming_words]
            alarming_words = list(set(alarming_words))  # Remove duplicates

            # Determine risk level
            if score <= 3:
                risk_level = 'low'
            elif score <= 5:
                risk_level = 'medium'
            elif score <= 7:
                risk_level = 'high'
            else:
                risk_level = 'critical'

            return {
                'score': round(score, 2),
                'probability': round(suicidal_prob, 4),
                'alarming_words': alarming_words,
                'risk_level': risk_level
            }
            
        except Exception as e:
            print(f"Error in suicide detection prediction: {e}")
            return {
                'score': 1.0,
                'probability': 0.0,
                'alarming_words': [],
                'risk_level': 'low',
                'error': str(e)
            }

# Global instance for reuse
_suicide_detector = None

def get_suicide_detector():
    """
    Get or create a global suicide detector instance
    """
    global _suicide_detector
    if _suicide_detector is None:
        _suicide_detector = SuicideDetector()
    return _suicide_detector

def analyze_text_for_suicide_risk(text):
    """
    Convenience function to analyze text for suicide risk
    """
    detector = get_suicide_detector()
    return detector.predict_suicidal_intent(text)
