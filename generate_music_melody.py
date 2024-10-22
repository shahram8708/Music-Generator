from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torch
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

model.eval()

prompt = "A calm and soothing melody with soft piano and birds chirping in the background."

inputs = processor(text=[prompt], return_tensors="pt")

with torch.no_grad():
    generated_audio = model.generate(**inputs, do_sample=True, max_new_tokens=1024)

waveform = generated_audio[0].cpu()

sample_rate = 32000

output_file = "generated_music.wav"
torchaudio.save(output_file, waveform.unsqueeze(0), sample_rate)

print(f"Music has been generated and saved as {output_file}.")
