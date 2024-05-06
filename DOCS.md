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

![Task3](task3.png)

## Task 4: Use Ensemble Methods for Verification *advanced*

## Task 5: Implementing Anomaly Detection *basic*