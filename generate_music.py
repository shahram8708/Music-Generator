from transformers import MusicgenProcessor, MusicgenForConditionalGeneration
import torch
import torchaudio

# Load the MusicGen processor and model (currently using the "large" model).
# Note: You can also use the "medium" or "small" models instead of "large"
# based on your system's available resources. Simply replace "musicgen-large"
# with "musicgen-medium" or "musicgen-small" in both lines below to reduce
# memory usage and computational load.

processor = MusicgenProcessor.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")

model.eval()

prompt = "A calm and soothing melody with soft piano and birds chirping in the background."

inputs = processor(text=[prompt], return_tensors="pt")

with torch.no_grad():
    generated_audio = model.generate(**inputs, max_new_tokens=1000)

waveform = generated_audio.squeeze(0)

sample_rate = 32000

output_file = "generated_music.wav"
torchaudio.save(output_file, waveform, sample_rate)

print(f"Music has been generated and saved as {output_file}. You can now download it.")
