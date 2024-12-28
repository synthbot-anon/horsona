# Speech Generation Sample

This sample demonstrates how to use the GPT-SoVITS text-to-speech system to generate speech from text input.

## Setup Instructions

### Setting up GPT-SoVITS TTS endpoint

1. Create a new folder called `gptsovits_data`

2. Download and unzip [these files](https://drive.google.com/file/d/106i6hQVDrUuULe_k8-MSi7wB4fW0X2Qx/view?usp=sharing) into the `gptsovits_data` folder. This contains:
   - Required pretrained models 
   - API config file
   - Sample voices in the required directory structure

3. From inside the `gptsovits_data` folder, run this command to start the GPT-SoVITS server:
   ```bash
   docker run --name gptsovits --rm -it -p 9880:9880 \
     -v "./tts-configs:/workspace/GPT_SoVITS/configs" \
     -v "./voices:/voices" \
     synthbot/gpt-sovits:v3 python api.py -a 0.0.0.0
   ```

### Setting up the Repository

1. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

### Running the Sample

1. Run the sample script:
   ```bash
   poetry run python src/main.py
   ```

This will start an interactive prompt where you can enter text to be converted to speech. The generated audio files will be saved in the current directory with incrementing numbers (e.g. `rarity_0.wav`, `rarity_1.wav`, etc).
