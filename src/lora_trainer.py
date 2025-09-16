from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def train_lora(model_name="mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

    # Example dataset (replace with faculty grading dataset later)
    dataset = load_dataset("imdb", split="train[:1%]")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    model.save_pretrained("./checkpoints/lora")
    print("✅ LoRA fine-tuning complete. Model saved at ./checkpoints/lora")

if __name__ == "__main__":
    train_lora()
