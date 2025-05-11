import os
from transformers import GPT2TokenizerFast

# Check if there's a local 'gpt2' directory
if os.path.exists("gpt2"):
    print("Warning: Local 'gpt2' directory exists. This may cause conflicts.")
    # Rename it temporarily
    os.rename("gpt2", "gpt2_backup")

# Now try loading again
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')