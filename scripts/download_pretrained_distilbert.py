"""Download DistilBERT pretrained model and tokenizer for offline server use."""
import os

from transformers import DistilBertModel, DistilBertTokenizerFast

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_distilbert")
os.makedirs(output_dir, exist_ok=True)

print("Downloading DistilBERT-base-uncased model...")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

print("Downloading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to {output_dir}")

print("Verifying local load...")
model2 = DistilBertModel.from_pretrained(output_dir)
tok2 = DistilBertTokenizerFast.from_pretrained(output_dir)
print(f"Model config: hidden_size={model2.config.hidden_size}, n_layers={model2.config.n_layers}, n_heads={model2.config.n_heads}")
print(f"Tokenizer vocab_size={tok2.vocab_size}, pad_token_id={tok2.pad_token_id}")
print("Done!")