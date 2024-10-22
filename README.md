# MusicGen Conditional Generation

This repository contains two scripts that use Facebook's MusicGen model to generate music based on text prompts. MusicGen is a Transformer model capable of generating music conditioned on text descriptions.

## Overview

This repository features two different scripts:
1. **generate_music.py**: Uses the **MusicGen Large** model for music generation based on a text prompt.
2. **generate_music_melody.py**: Uses the **MusicGen Melody** model, which is optimized for melody-conditioned music generation, also based on a text prompt.

Both scripts use PyTorch and the HuggingFace `transformers` library to load the models, process the text, and generate music. The generated music is saved as a `.wav` file.

### Model Size Variants

You can adjust the size of the model based on your system’s resources:
- The default script uses the **MusicGen Large** model.
- To reduce memory usage or computational load, you can switch to smaller models like **MusicGen Medium** or **MusicGen Small**.

Simply replace `"facebook/musicgen-large"` with:
- `"facebook/musicgen-medium"`
- `"facebook/musicgen-small"`

in both the `processor` and `model` loading lines of the code to reduce resource consumption.
