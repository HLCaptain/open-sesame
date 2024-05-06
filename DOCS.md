# Open Sesame UX Lab Docs

Neptun: BL6ADS
Author: Balázs Püspök-Kiss

## Task 1: Enhance Error Handling *basic*

Implemented error handling for the following cases:

- File not provided
- Audio processing failed
- Overall file handling error

```python
@app.route('/verify', methods=['POST'])
def verify_speaker():
    try:
        incoming_file = request.files['file']
        if not incoming_file:
            return jsonify({'error': 'No file provided'}), 400

        temp_filename = secure_filename(incoming_file.filename)
        temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        incoming_file.save(temp_filepath)

        # Handle potential errors during audio processing
        try:
            converted_filename = f"converted_{temp_filename}"
            converted_filepath = os.path.join(UPLOAD_FOLDER, converted_filename)
            conversion_thread = threading.Thread(target=convert_audio, args=(temp_filepath, converted_filepath))
            conversion_thread.start()
            conversion_thread.join()  # Wait for the conversion to complete before proceeding
        except Exception as e:
            return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'File handling error: {str(e)}'}), 500

    # ...
```

## Task 2: Add Multithreading for Handling File Conversion *basic*

Implemented multithreading for file conversion.

```python
import threading

def convert_audio(source_path, target_path):
    audio = AudioSegment.from_file_using_temporary_files(source_path).set_frame_rate(16000).set_channels(1)
    audio.export(target_path, format="wav")

@app.route('/verify', methods=['POST'])
def verify_speaker():
    try:
        incoming_file = request.files['file']
        if not incoming_file:
            return jsonify({'error': 'No file provided'}), 400

        temp_filename = secure_filename(incoming_file.filename)
        temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        incoming_file.save(temp_filepath)

        # Handle potential errors during audio processing
        try:
            converted_filename = f"converted_{temp_filename}"
            converted_filepath = os.path.join(UPLOAD_FOLDER, converted_filename)
            conversion_thread = threading.Thread(target=convert_audio, args=(temp_filepath, converted_filepath))
            conversion_thread.start()
            conversion_thread.join()  # Wait for the conversion to complete before proceeding
        except Exception as e:
            return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'File handling error: {str(e)}'}), 500

    # ...
```

## Task 3: Implement Dynamic Thresholding for Decision Making *intermediate*

Added new Table for storing past scores for users:

```python
class VerificationScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Float, nullable=False)
```

Verification now looks like this:

```python
@app.route('/verify', methods=['POST'])
def verify_speaker():
    # Audio loaded

    highest_score = -1
    verified_email = None
    verified_name = None

    for user in User.query.all():
        threshold = calculate_dynamic_threshold(user.id)
        # Verification logic using the dynamic threshold...
        email_hash = hashlib.sha256(user.email.encode('utf-8')).hexdigest()
        for i in range(1, 2):
            expected_filename = f"{email_hash}_{i}.wav"
            user_file_path = os.path.join(app.config['UPLOAD_FOLDER'], expected_filename)

            if user_file_path:
                score, prediction = verification.verify_files(converted_filepath, user_file_path)
                score = score.item()  # Convert tensor to a Python float

                if score > threshold and score > highest_score:
                    highest_score = score
                    verified_email = user.email if prediction else None
                    verified_name = user.name if prediction else None

    os.remove(temp_filepath)
    os.remove(converted_filepath)
```

Dynamic thresholding can improve the accuracy of the verification process by adjusting the threshold based on the user's personal history of verification scores, depending on its setup, microphone quality and speeking consistency. If the user's speech is low quality or does not match well with previous recordings, the threshold will be lower (but still acceptable). If the user's speech is high quality and matches well with previous recordings, the threshold will be higher, making the verification process more strict.

![Task3](task3.png)

## Task 4: Use Ensemble Methods for Verification *advanced*

Using multiple models for verification:

- `speechbrain/spkrec-ecapa-voxceleb`: returns 0 or 1 (false or true) for same speaker.
- `microsoft/wavlm-base-plus-sv`: returns a latent space vector for an audio file. Using cosine similarity to compare vectors.

```python
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from scipy.spatial.distance import cosine
import librosa

# Initialize and use Microsoft model to compute cosine similarity
class MicrosoftModel:
    def __init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

    def predict(self, audio_path1, audio_path2):
        ref, ref_rate = librosa.load(audio_path1, sr=16000)
        deg, deg_rate = librosa.load(audio_path2, sr=16000)
        audio = [ref, deg]
        inputs = self.feature_extractor(audio, padding=True, return_tensors="pt")
        embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embeddings[0], embeddings[1])
        return similarity

# Initialize and use SpeechBrain model to compute verification score
class SpeechBrainModel:
    def __init__(self):
        self.model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

    def predict(self, audio_path1, audio_path2):
        score, prediction = self.model.verify_files(audio_path1, audio_path2)
        return score.item()

ensemble_models = [SpeechBrainModel(), MicrosoftModel()]

# Verify with ensemble and average the scores
def verify_with_ensemble(audio_path1, audio_path2):
    scores = [model.predict(audio_path1, audio_path2) for model in ensemble_models]
    final_score = sum(scores) / len(scores)  # Average the scores
    return final_score
```

This approach can improve the accuracy of verification due to using multiple models as a basis.

## Task 5: Implementing Anomaly Detection *basic*

Implemented simple anomaly detection for when the threshold becomes too low (accepting a speaker that should not be accepted).

```python
def detect_anomalies(scores):
    """Detect anomalies in verification scores which might indicate a security breach."""
    predetermined_threshold = 0.2
    if np.std(scores.detach().numpy()) > predetermined_threshold:
        return True
    else:
        return False
```
