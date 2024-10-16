import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import warnings
warnings.filterwarnings("ignore")

import torch
import time
import datetime
from diffusers import DiffusionPipeline

import nltk
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet152_Weights
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

# Build Vocabulary
class Vocab(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0
 
    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]
 
    def __len__(self):
        return len(self.w2i)
    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1
            
class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNModel, self).__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        module_list = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_module = nn.Sequential(*module_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)
        
    def forward(self, input_images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        final_features = self.batch_norm(self.linear_layer(resnet_features))
        return final_features
    
class LSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        """Set the hyper-parameters and build the layers."""
        super(LSTMModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len
        
    def forward(self, input_features, capts, lens):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding_layer(caps)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True) 
        hidden_variables, _ = self.lstm_layer(lstm_input)
        model_outputs = self.linear_layer(hidden_variables[0])
        return model_outputs
    
    def sample(self, input_features, lstm_states=None):
        """Generate captions for given image features using greedy search."""
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs, lstm_states) # hiddens: (batch_size, 1, hidden_size)
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted_outputs = model_outputs.max(1)                               # predicted: (batch_size)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embedding_layer(predicted_outputs)                     # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(1)                                    # inputs: (batch_size, 1, embed_size)
        sampled_indices = torch.stack(sampled_indices, 1)                             # sampled_ids: (batch_size, max_seq_length)
        return sampled_indices
            
def load_image(image_file_path, transform=None):
    img = Image.open(image_file_path).convert('RGB')
    img = img.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        img = transform(img).unsqueeze(0)
    return img
 
def describe_image(image, device):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open('data_dir/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    # Build models
    encoder_model = CNNModel(256).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder_model = LSTMModel(256, 512, len(vocabulary), 1)
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)

    # Load the trained model parameters
    encoder_model.load_state_dict(torch.load('models_dir/encoder-2-3000.ckpt', weights_only=True))
    decoder_model.load_state_dict(torch.load('models_dir/decoder-2-3000.ckpt', weights_only=True))
    
    # Prepare an image
    img = load_image(image, transform)
    img_tensor = img.to(device)

    # Generate an caption from the image
    feat = encoder_model(img_tensor)
    sampled_indices = decoder_model.sample(feat)
    sampled_indices = sampled_indices[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    predicted_caption = []
    for token_index in sampled_indices:
        word = vocabulary.i2w[token_index]
        predicted_caption.append(word)
        if word == '<end>':
            break
        
    # Strip start and end sentence marker.
    predicted_caption = predicted_caption[1:-2]     # Remove "." 
    predicted_sentence = ' '.join(predicted_caption)
    print ("Prompt : ", predicted_sentence)
    return predicted_sentence

def generate_image(prompt, device):
    pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                            torch_dtype=torch.float16, 
                                            use_safetensors=True, 
                                            variant="fp16")
    pipe.to(device)
    image = pipe(prompt=prompt).images[0]
    return image

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using... : ", device)

    # Changed from "punkt" to "punkt_tab"
    nltk.download('punkt_tab')
    
    start = time.time()
    #prompt = describe_image("sample01.jpg", device)
    #prompt = describe_image("sample02.jpg", device)
    #prompt = describe_image("sample03.jpg", device)
    #prompt = describe_image("sample04.jpg", device)
    #image = generate_image(prompt, device)
    #prompt = "A woman wearing a swimsuit is working on the street"
    prompt = "sunset in the beach"
    image = generate_image(prompt, device)
    end = time.time()
    seconds = end - start
    result = str(datetime.timedelta(seconds=seconds)).split(".")
    print("Total elapsed time : ", result[0])
    
    # Save the gneerated image.
    image.save(f"{prompt}.png")
    
if __name__ == '__main__':
    main()
