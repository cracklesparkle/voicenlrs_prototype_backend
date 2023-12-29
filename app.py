from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import io
import torchaudio
from omegaconf import OmegaConf
import soundfile

app = Flask(__name__)
CORS(app)
# Download models.yml file
torch.hub.download_url_to_file(
    'https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
    'latest_silero_models.yml',
    progress=False
)

# Load models.yml file
models = OmegaConf.load('latest_silero_models.yml')

# Get available languages and models
#available_languages = list(models.tts_models.keys())
#print(f'Available languages {available_languages}')

# Set language and model_id based on your requirement
language = 'cyrillic'
model_id = 'v4_cyrillic'
device = torch.device('cpu')

# Load the Silero TTS model
model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language=language,
    speaker=model_id
)
model.to(device)  # gpu or cpu

# Define the route for TTS synthesis
@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text', '')

    # Synthesize speech
    sample_rate = 48000
    speaker = 'b_sah'
    put_accent = True
    put_yo = True
    audio = model.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate,
        put_accent=put_accent,
        put_yo=put_yo
    )

    # Check if the audio tensor has multiple channels
    if audio.ndim == 2:
        audio = audio.mean(dim=0)  # Average multiple channels

    # Convert audio tensor to bytes
    buffer = io.BytesIO()
    soundfile.write(buffer, audio.numpy(), sample_rate, format='wav')
    audio_bytes = buffer.getvalue()

    # Return audio bytes as part of the Response
    return Response(audio_bytes, content_type="audio/wav")


if __name__ == '__main__':
    app.run(debug=True)
