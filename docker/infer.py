import os
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Configuration for model
model_size = "tiny"  # You can change this to "base", "small", "medium", "large-v2", "large-v3" etc.
device_type = "cuda"  # "cuda" or "cpu"
compute_type_model = "float16"  # "float16", "int8_float16", "int8" for GPU; "int8", "float32" for CPU

print(f"Loading Whisper model: {model_size} on {device_type} with {compute_type_model}")
try:
    model = WhisperModel(model_size, device=device_type, compute_type=compute_type_model)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or exit if model loading fails critically
    # For now, we'll let it try to run, but Flask will error out if model is None
    model = None


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_file"]

    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file to a temporary file
    # faster-whisper can take a file path
    try:
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, audio_file.filename or "audio_upload")
        audio_file.save(temp_audio_path)
        print(f"Audio file saved to temporary path: {temp_audio_path}")

        # Perform transcription
        segments, info = model.transcribe(temp_audio_path, word_timestamps=True)

        print(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )

        transcription_text = []
        for segment in segments:
            transcription_text.append(
                "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            )
            # To get word timestamps:
            # for word in segment.words:
            #     print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

        full_transcription = "\n".join(transcription_text)
        print(f"Transcription: {full_transcription}")

        # Clean up the temporary file and directory
        os.remove(temp_audio_path)
        os.rmdir(temp_dir)
        print("Temporary file and directory removed.")

        return jsonify(
            {
                "language": info.language,
                "language_probability": info.language_probability,
                "transcription": full_transcription,
            }
        )

    except Exception as e:
        print(f"Error during transcription: {e}")
        # Clean up if an error occurs after file creation
        if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    # For development, Flask's dev server is fine.
    # For production, use a WSGI server like Gunicorn or uWSGI.
    app.run(host="0.0.0.0", port=5000, debug=False)
