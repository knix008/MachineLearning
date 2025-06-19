# Transformer Model Training with Penn Treebank Dataset

This project implements a complete transformer model training pipeline with a Gradio web interface, using the Penn Treebank dataset without requiring torchtext.

## Features

- 🤖 **Modern Transformer Architecture**: Implements a transformer model with positional encoding and causal masking
- 📊 **Penn Treebank Dataset**: Uses the Penn Treebank dataset without torchtext dependency
- 🎨 **Gradio Web Interface**: User-friendly web interface for training and text generation
- 📈 **Training Visualization**: Real-time plotting of training and validation loss
- 🔧 **Customizable Parameters**: Adjustable model architecture and training parameters
- 📝 **Text Generation**: Interactive text generation with temperature control
- 💾 **Model Persistence**: Automatic saving of the best model during training

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Penn Treebank dataset in the `ptbdataset/` folder:
   - `ptbdataset/ptb.train.txt`
   - `ptbdataset/ptb.valid.txt`
   - `ptbdataset/ptb.test.txt`

## Usage

### Running the Gradio Interface

```bash
python GPT2GradioExample03.py
```

This will launch a web interface with three main tabs:

### 1. Training Tab
Configure and start training the transformer model:

**Training Parameters:**
- **Epochs**: Number of training epochs (1-50)
- **Batch Size**: Training batch size (8-128)
- **Sequence Length**: Length of input sequences (16-256)
- **Vocabulary Size**: Size of vocabulary (1000-20000)

**Model Architecture:**
- **Model Dimension**: Hidden dimension size (64-512)
- **Number of Heads**: Number of attention heads (2-16)
- **Number of Layers**: Number of transformer layers (2-12)
- **Learning Rate**: Training learning rate (1e-5 to 1e-2)

**Generation Parameters:**
- **Temperature**: Sampling temperature for text generation (0.1-2.0)
- **Max Generation Length**: Maximum length of generated text (10-200)

### 2. Text Generation Tab
Generate text using a trained model:

- **Input Prompt**: Enter your starting text
- **Temperature**: Control randomness in generation
- **Max Length**: Maximum length of generated text

### 3. Model Info Tab
Information about the implementation and features.

## Model Architecture

The transformer model includes:

- **Word Embeddings**: Learnable word embeddings
- **Positional Encoding**: Learnable positional encodings
- **Multi-Head Self-Attention**: Causal attention for language modeling
- **Feed-Forward Networks**: Position-wise feed-forward layers
- **Layer Normalization**: For stable training
- **Dropout**: For regularization

## Training Process

1. **Data Loading**: Loads Penn Treebank dataset without torchtext
2. **Vocabulary Building**: Creates vocabulary from training data
3. **Dataset Creation**: Converts text to sequences for training
4. **Model Training**: Trains transformer with gradient clipping
5. **Validation**: Evaluates on validation set
6. **Model Saving**: Saves best model based on validation loss
7. **Text Generation**: Generates sample text with trained model

## File Structure

```
├── GPT2GradioExample03.py    # Main training script with Gradio interface
├── requirements.txt          # Python dependencies
├── README_Transformer_Training.md  # This file
├── ptbdataset/              # Penn Treebank dataset
│   ├── ptb.train.txt
│   ├── ptb.valid.txt
│   └── ptb.test.txt
└── best_transformer_model.pth  # Saved model (created after training)
```

## Key Components

### PTBDataset Class
Custom dataset class that loads Penn Treebank data without torchtext:
- Reads text files directly
- Converts words to indices
- Creates sliding window sequences for training

### TransformerModel Class
Modern transformer implementation:
- Learnable positional encodings
- Causal attention masking for language modeling
- Configurable architecture parameters

### TransformerTrainer Class
Training wrapper with:
- Epoch-based training loop
- Validation during training
- Gradient clipping
- Model checkpointing
- Training history tracking

### Gradio Interface
User-friendly web interface with:
- Interactive parameter sliders
- Real-time training progress
- Text generation interface
- Training visualization

## Example Usage

1. **Start Training**:
   - Set epochs to 10
   - Set batch size to 32
   - Set sequence length to 64
   - Click "Start Training"

2. **Generate Text**:
   - Enter prompt: "the company said"
   - Set temperature to 1.0
   - Click "Generate Text"

## Tips for Best Results

1. **Start Small**: Begin with smaller models (fewer layers, smaller dimensions)
2. **Monitor Loss**: Watch training and validation loss curves
3. **Adjust Learning Rate**: Lower learning rate if training is unstable
4. **Use GPU**: Training will be much faster with GPU acceleration
5. **Experiment with Temperature**: Lower temperature (0.5-0.8) for more focused text, higher (1.2-1.5) for more creative text

## Troubleshooting

**Dataset Not Found**: Ensure `ptbdataset/` folder exists with train and validation files.

**Out of Memory**: Reduce batch size, sequence length, or model dimensions.

**Training Too Slow**: Use GPU if available, reduce model size, or use smaller vocabulary.

**Poor Generation Quality**: Train for more epochs, adjust temperature, or increase model capacity.

## Technical Details

- **Loss Function**: Cross-entropy loss with padding token ignored
- **Optimizer**: AdamW with weight decay
- **Gradient Clipping**: Norm clipping at 1.0
- **Model Saving**: Saves best model based on validation loss
- **Text Generation**: Uses temperature sampling with causal masking

## License

This project is for educational purposes. The Penn Treebank dataset has its own license terms. 