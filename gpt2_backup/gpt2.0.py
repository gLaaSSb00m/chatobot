# Add this import at the top
import numpy as np
from datasets import load_dataset, DatasetDict
from IPython.display import display  # Import display for displaying DataFrames
import pandas as pd  # Import pandas for DataFrame operations
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, TrainingArguments, Trainer  # Import GPT2TokenizerFast, AutoModelForCausalLM, TrainingArguments, and Trainer

# [Keep all your existing imports and device setup]
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare dataset
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
selected_train_dataset = dataset['train'].select(list(range(100)))

# Split dataset
train_testvalid = selected_train_dataset.train_test_split(test_size=0.1)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
split_datasets = DatasetDict({
    'train': train_testvalid['train'],
    'validation': test_valid['train'],
    'test': test_valid['test']
})

# Formatting function
def format_example(example):
    return {'text': f"Question: {example['Question']} Answer: {example['Answer']} <|endoftext|>"}

# Apply formatting
formatted_datasets = split_datasets.map(format_example)

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir="./custom_cache")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    labels = np.array(inputs['input_ids'], dtype=np.int64)
    labels[labels == tokenizer.pad_token_id] = -100
    inputs['labels'] = labels.tolist()
    return inputs

# Tokenize all datasets
tokenized_datasets = formatted_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=['qtype', 'Question', 'Answer', 'text']  # Remove all original columns
)

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

training_args = TrainingArguments(
    output_dir="./gpt2-medquad-finetuned",
    eval_strategy="epoch",  # Changed from evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train and save
trainer.train()
model.save_pretrained("./gpt2-medquad-finetuned")
tokenizer.save_pretrained("./gpt2-medquad-finetuned")

# Evaluation
results_train = trainer.evaluate(tokenized_datasets["train"])
results_val = trainer.evaluate(tokenized_datasets["validation"])
results_test = trainer.evaluate(tokenized_datasets["test"])

# Display results
results_df = pd.DataFrame({
    "Dataset": ["Training", "Validation", "Testing"],
    "Loss": [results_train['eval_loss'], results_val['eval_loss'], results_test['eval_loss']],
    # Add other metrics as needed
})
display(results_df)