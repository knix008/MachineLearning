import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# 1. PTB 데이터셋 로드 및 전처리
def load_ptb_data():
    dataset = load_dataset("ptb_text_only")
    train_texts = "\n".join(dataset["train"]["sentence"])
    val_texts = "\n".join(dataset["validation"]["sentence"])
    with open("ptb_train.txt", "w", encoding="utf-8") as f:
        f.write(train_texts)
    with open("ptb_val.txt", "w", encoding="utf-8") as f:
        f.write(val_texts)
    return "Penn Tree Bank 데이터셋 로드 완료!"

# 2. 데이터셋 및 데이터 콜레이터 생성
def get_dataset(tokenizer, file_path, block_size=128):
    from transformers import TextDataset
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

# 3. 모델 훈련 함수
def train_gpt2_ptb(epochs, batch_size):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # pad와 eos 토큰 추가
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'eos_token': ''})  # idempotent

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # ---- Set loss_type in model config ----
    model.config.loss_type = "ForCausalLMLoss"
    # ---------------------------------------

    train_dataset = get_dataset(tokenizer, "ptb_train.txt")
    val_dataset = get_dataset(tokenizer, "ptb_val.txt")
    data_collator = get_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir="./gpt2-ptb-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=int(epochs),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        #evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    model.save_pretrained("./gpt2-ptb-finetuned")
    tokenizer.save_pretrained("./gpt2-ptb-finetuned")
    return f"Train loss: {eval_results['eval_loss']:.4f})"

# 4. 텍스트 생성 함수 (attention mask 사용)
def generate_text_ptb(prompt, max_length):
    model = GPT2LMHeadModel.from_pretrained("./gpt2-ptb-finetuned")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-ptb-finetuned")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    encoded = tokenizer(prompt, return_tensors="pt", padding="longest")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    sample_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_length=int(max_length),
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(sample_output[0], skip_special_tokens=True)

# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("## GPT-2 Penn Tree Bank 훈련 및 텍스트 생성")
    with gr.Tab("PTB 데이터셋 준비"):
        ptb_btn = gr.Button("PTB 데이터셋 준비")
        ptb_output = gr.Textbox(label="데이터셋 준비 결과")
        ptb_btn.click(
            load_ptb_data,
            inputs=[],
            outputs=ptb_output
        )
    with gr.Tab("모델 훈련"):
        epochs = gr.Number(label="에폭 (epochs)", value=1)
        batch_size = gr.Number(label="배치 크기 (batch size)", value=2)
        train_btn = gr.Button("훈련 시작")
        train_output = gr.Textbox(label="훈련 결과")
        train_btn.click(
            lambda e, b: train_gpt2_ptb(e, b),
            inputs=[epochs, batch_size],
            outputs=train_output
        )
    with gr.Tab("텍스트 생성"):
        prompt = gr.Textbox(label="프롬프트")
        max_length = gr.Number(label="최대 길이", value=50)
        gen_btn = gr.Button("텍스트 생성")
        gen_output = gr.Textbox(label="생성된 텍스트")
        gen_btn.click(
            generate_text_ptb,
            inputs=[prompt, max_length],
            outputs=gen_output
        )

if __name__ == "__main__":
    demo.launch()