import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)
import nltk
from nltk.corpus import gutenberg
import os


def load_gutenberg_corpus():
    # Download NLTK data
    nltk.download("gutenberg")
    # Prepare data: use a sample from NLTK Gutenberg corpus
    text = gutenberg.raw("austen-emma.txt")[
        :100_000
    ]  # Use only first 100k chars for demo

    # Save to file (HuggingFace Trainer expects text files)
    os.makedirs("data", exist_ok=True)
    with open("data/nltk_text.txt", "w", encoding="utf-8") as f:
        f.write(text)


# Create dataset and data collator
def get_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
        overwrite_cache=True,
    )


def main():
    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Load the dataset
    load_gutenberg_corpus()
    train_dataset = get_dataset("data/nltk_text.txt", tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=50,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Fine-tune
    print("> Starting training...")
    trainer.train()

    # Save the model
    print("> Saving the model...")
    model.save_pretrained("./trained_gpt2_nltk")
    tokenizer.save_pretrained("./trained_gpt2_nltk")

    # Generate text from the fine-tuned model
    print("> Generating text...")
    prompt = "She was not struck by anything remarkably clever"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,  # Use EOS token as padding
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("> Generated text:\n", generated_text)


if __name__ == "__main__":
    # Run the training script
    main()
