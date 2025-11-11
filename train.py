import torch
import os
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Paths
# Fix in train.py
base_model_name = "reedmayhew/Grok-3-gemma3-4B-distilled"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False, token=os.getenv("HF_TOKEN"))
inputs = tokenizer("Chill test...", return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
# Load tokenizer with fixes`
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False, token=os.getenv("HF_TOKEN"))

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_4bit=True, device_map="auto")

# Load or create PEFT model
model = PeftModel.from_pretrained(base_model, finetuned_path, is_trainable=True) if os.path.exists(finetuned_path) else base_model

if not hasattr(model, 'peft_config') or not model.peft_config:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files=data_path, split="train")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
args = TrainingArguments(
    output_dir=r"X:\grokshit\ttl\30d\export_data\40b8b6ba-1e88-478a-b457-1c3a18329113\grok-finetuned-resumed",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    optim="paged_adamw_8bit",
    resume_from_checkpoint=finetuned_path
)

# Train
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train(resume_from_checkpoint=True if os.path.exists(finetuned_path) else False)

# Save
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# Test
inputs = tokenizer("Chill test with my chat history", return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))