import os
import warnings
import torch
from transformers import pipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
warnings.filterwarnings("ignore")

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using... : ", device)
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-small", device=device)
    music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})
    scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    prompt = "80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    audio_values = model.generate(**inputs, max_new_tokens=256)
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

if __name__ == "__main__":
    main()