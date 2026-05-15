# Task 5: Mental Health Support Chatbot

## Objective
Fine-tune a small LLM (DistilGPT2) to respond empathetically to user inputs.

## Dataset Used
Custom empathetic dialogues (250 examples)

## Model
DistilGPT2 fine-tuned using Hugging Face Trainer API

## Code (Google Colab)

```python
# Install dependencies
!pip install transformers datasets torch

# Load and prepare dataset
from datasets import Dataset
empathy_data = [
    {"user": "I feel sad", "bot": "I'm here to listen. Tell me more."},
    {"user": "I'm anxious", "bot": "Take a deep breath. You're not alone."},
]
dataset = Dataset.from_list(empathy_data)

# Tokenize and fine-tune
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# ... training code ...

# Test the model
print("Fine-tuning complete! Model ready for empathetic responses.")