from flask import Flask, request, render_template
import whisper

app = Flask(__name__)

# Load your model
model = whisper.load_model("medium")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['audio']
    # Process the audio file with your model
    result = model.transcribe(audio_file)
    return result['text']

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
