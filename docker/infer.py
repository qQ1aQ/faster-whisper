import os
import tempfile

from faster_whisper import WhisperModel

from flask import Flask, jsonify, request


app = Flask(__name__)

# Load the Whisper model once when the application starts
model_size = "tiny"
model = WhisperModel(model_size, device="cuda", compute_type="float16")


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    # Save the audio file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio_file.save(tmp_file.name)
        temp_audio_path = tmp_file.name

    try:
        # Transcribe the audio file
        segments, info = model.transcribe(temp_audio_path, word_timestamps=True)
        transcription = []
        for segment in segments:
            transcription.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )

        response_data = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": transcription,
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up the temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == "__main__":
    # Run the Flask app on port 5000
    app.run(host="0.0.0.0", port=5000)
