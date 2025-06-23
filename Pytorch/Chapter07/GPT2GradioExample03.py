import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import os
import json
from collections import Counter
import warnings
from typing import List, Tuple, Dict
import time

# Suppress warnings
warnings.filterwarnings(action='ignore')

# Penn Treebank Dataset Loader (without torchtext)
class PTBDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int, word2idx: Dict[str, int], idx2word: List[str]):
        self.seq_len = seq_len
        self.word2idx = word2idx
        self.idx2word = idx2word
        
        # Read and preprocess data
        with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Split into words and convert to indices
        words = text.replace('\n', ' <eos> ').split()
        self.data = [word2idx.get(word, word2idx['<UNK>']) for word in words if word in word2idx]
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

def build_vocab(train_path: str, vocab_size: int = 10000) -> Tuple[Dict[str, int], List[str]]:
    """Build vocabulary from training data"""
    with open(train_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    words = text.replace('\n', ' <eos> ').split()
    word_counts = Counter(words)
    
    # Get most common words
    most_common = word_counts.most_common(vocab_size - 3)  # -3 for special tokens
    
    # Create vocabulary with special tokens
    idx2word = ['<PAD>', '<UNK>', '<EOS>'] + [word for word, _ in most_common]
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    
    return word2idx, idx2word

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 1024, 
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.pos_encoding, std=0.02)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # Embeddings + positional encoding
        embeddings = self.embedding(x) * (self.d_model ** 0.5)
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :]
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for causal language modeling
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        
        # Transformer forward pass
        output = self.transformer(embeddings, mask=mask)
        
        # Output projection
        logits = self.output_layer(output)
        
        return logits

class TransformerTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(x)
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)
                
                loss = criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              optimizer: optim.Optimizer, criterion: nn.Module, 
              epochs: int, save_path: str = None) -> Dict[str, List[float]]:
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss = self.validate(val_loader, criterion)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'training_history': self.training_history
                }, save_path)
                print(f"Model saved to {save_path}")
        
        return self.training_history

def generate_text(model: nn.Module, prompt: str, word2idx: Dict[str, int], 
                 idx2word: List[str], max_length: int = 100, 
                 temperature: float = 1.0, device: str = 'cpu') -> str:
    """Generate text using the trained model"""
    model.eval()
    
    # Tokenize prompt
    words = prompt.split()
    input_ids = [word2idx.get(word, word2idx['<UNK>']) for word in words]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            x = torch.tensor([input_ids[-model.max_seq_len:]], dtype=torch.long).to(device)
            
            # Get predictions
            logits = model(x)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to sequence
            input_ids.append(next_token)
            
            # Stop if EOS token
            if next_token == word2idx['<EOS>']:
                break
    
    # Convert back to text
    generated_words = [idx2word[idx] for idx in input_ids]
    return ' '.join(generated_words)

def plot_training_history(history: Dict[str, List[float]]) -> str:
    """Create and save training history plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = 'training_history.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

# Gradio Interface
def create_gradio_interface():
    """Create Gradio interface for transformer training"""
    
    def train_transformer(
        epochs: int,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        learning_rate: float,
        temperature: float,
        max_gen_length: int
    ):
        try:
            # Check if data exists
            train_path = "./ptbdataset/ptb.train.txt"
            valid_path = "./ptbdataset/ptb.valid.txt"
            
            if not os.path.exists(train_path) or not os.path.exists(valid_path):
                return "Error: Penn Treebank dataset not found. Please ensure ptbdataset folder exists with train and validation files.", None, None
            
            # Device setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Build vocabulary
            print("Building vocabulary...")
            word2idx, idx2word = build_vocab(train_path, vocab_size)
            
            # Create datasets
            print("Creating datasets...")
            train_dataset = PTBDataset(train_path, seq_len, word2idx, idx2word)
            val_dataset = PTBDataset(valid_path, seq_len, word2idx, idx2word)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            print("Initializing model...")
            model = TransformerModel(
                vocab_size=len(word2idx),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_seq_len=seq_len
            )
            
            # Setup training
            trainer = TransformerTrainer(model, device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
            
            # Training
            print("Starting training...")
            history = trainer.train(
                train_loader, val_loader, optimizer, criterion, 
                epochs=epochs, save_path='best_transformer_model.pth'
            )
            
            # Generate sample text
            print("Generating sample text...")
            sample_prompts = [
                "the company said",
                "in the beginning",
                "the government announced",
                "according to the report"
            ]
            
            generated_texts = []
            for prompt in sample_prompts:
                generated = generate_text(
                    model, prompt, word2idx, idx2word, 
                    max_length=max_gen_length, temperature=temperature, device=device
                )
                generated_texts.append(f"Prompt: {prompt}\nGenerated: {generated}\n")
            
            # Create training plot
            plot_path = plot_training_history(history)
            
            # Prepare results
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            
            results_summary = f"""
                Training completed successfully!

                Final Training Loss: {final_train_loss:.4f}
                Final Validation Loss: {final_val_loss:.4f}
                Vocabulary Size: {len(word2idx)}
                Model saved to: best_transformer_model.pth

                Sample Generations:
                {''.join(generated_texts)}
                """
            
            return results_summary, plot_path, history
            
        except Exception as e:
            return f"Error during training: {str(e)}", None, None
    
    def generate_text_interface(prompt: str, temperature: float, max_length: int):
        try:
            # Load model
            if not os.path.exists('best_transformer_model.pth'):
                return "Error: No trained model found. Please train the model first."
            
            # Load vocabulary (you might need to save this separately)
            train_path = "./ptbdataset/ptb.train.txt"
            word2idx, idx2word = build_vocab(train_path, 10000)
            
            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TransformerModel(vocab_size=len(word2idx))
            
            checkpoint = torch.load('best_transformer_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Generate text
            generated = generate_text(
                model, prompt, word2idx, idx2word, 
                max_length=max_length, temperature=temperature, device=device
            )
            
            return f"Generated text:\n{generated}"
            
        except Exception as e:
            return f"Error during generation: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="Transformer Training with Penn Treebank", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Transformer Model Training with Penn Treebank")
        gr.Markdown("Train a transformer model on the Penn Treebank dataset without using torchtext")
        
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Training Parameters")
                    epochs = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Epochs")
                    batch_size = gr.Slider(minimum=8, maximum=128, value=32, step=8, label="Batch Size")
                    seq_len = gr.Slider(minimum=16, maximum=256, value=64, step=16, label="Sequence Length")
                    vocab_size = gr.Slider(minimum=1000, maximum=20000, value=10000, step=1000, label="Vocabulary Size")
                    
                with gr.Column():
                    gr.Markdown("### Model Architecture")
                    d_model = gr.Slider(minimum=64, maximum=512, value=256, step=64, label="Model Dimension")
                    nhead = gr.Slider(minimum=2, maximum=16, value=8, step=2, label="Number of Heads")
                    num_layers = gr.Slider(minimum=2, maximum=12, value=6, step=1, label="Number of Layers")
                    learning_rate = gr.Slider(minimum=1e-5, maximum=1e-2, value=1e-3, step=1e-5, label="Learning Rate")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Generation Parameters")
                    temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    max_gen_length = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Max Generation Length")
                
                with gr.Column():
                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
            with gr.Row():
                output_text = gr.Textbox(label="Training Results", lines=10)
                training_plot = gr.Image(label="Training History")
            
            train_btn.click(
                fn=train_transformer,
                inputs=[epochs, batch_size, seq_len, vocab_size, d_model, nhead, num_layers, 
                       learning_rate, temperature, max_gen_length],
                outputs=[output_text, training_plot]
            )
        
        with gr.Tab("Text Generation"):
            gr.Markdown("### Generate Text with Trained Model")
            
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Input Prompt", 
                        placeholder="Enter your prompt here...",
                        value="the company said",
                        lines=3
                    )
                    gen_temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                    gen_max_length = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Max Length")
                    generate_btn = gr.Button("✨ Generate Text", variant="primary")
                
                with gr.Column():
                    generated_output = gr.Textbox(label="Generated Text", lines=10)
            
            generate_btn.click(
                fn=generate_text_interface,
                inputs=[prompt_input, gen_temperature, gen_max_length],
                outputs=[generated_output]
            )
        
        with gr.Tab("Model Info"):
            gr.Markdown("### About This Implementation")
            gr.Markdown("""
            This transformer model training interface features:
            
            - **Penn Treebank Dataset**: Uses the Penn Treebank dataset without torchtext dependency
            - **Custom DataLoader**: Implements custom dataset and dataloader for efficient training
            - **Transformer Architecture**: Modern transformer with positional encoding and causal masking
            - **Gradio Interface**: User-friendly web interface for training and text generation
            - **Training Visualization**: Real-time plotting of training and validation loss
            - **Text Generation**: Interactive text generation with temperature control
            
            The model uses causal language modeling to predict the next word given previous words.
            """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)

