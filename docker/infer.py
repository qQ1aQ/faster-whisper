import os
import tempfile
import traceback  # Moved to top with other stdlib imports

from flask import Flask, jsonify, request
from flask_cors import CORS  # New third-party import
from faster_whisper import WhisperModel


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins


# Configuration for model
MODEL_SIZE = "tiny"  # Constant for model size
DEVICE_TYPE = "cuda"  # "cuda" or "cpu"
COMPUTE_TYPE_MODEL = (
    "float16"  # "float16", "int8_float16", "int8" for GPU; "int8", "float32" for CPU
)

print(
    f"Loading Whisper model: {MODEL_SIZE} "
    f"on {DEVICE_TYPE} with {COMPUTE_TYPE_MODEL}"
)
try:
    model = WhisperModel(
        MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE_MODEL
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_file"]

    if not audio_file.filename:  # Simpler check for empty filename
        return jsonify({"error": "No selected file"}), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = audio_file.filename
            temp_audio_path = os.path.join(temp_dir, filename)
            audio_file.save(temp_audio_path)
            print(f"Audio file saved to temporary path: {temp_audio_path}")

            if model is None:  # Should be redundant if initial check is robust
                print(
                    "Model became None before transcription"
                )  # Corrected line length and comment
                return jsonify({"error": "Model not available for transcription"}), 500

            segments, info = model.transcribe(temp_audio_path, word_timestamps=True)

            print(
                f"Detected language '{info.language}' "
                f"with probability {info.language_probability:.2f}"
            )

            transcription_text_parts = []
            for segment in segments:
                transcription_text_parts.append(
                    f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                )

            full_transcription = "\n".join(transcription_text_parts)
            print(f"Transcription: {full_transcription}")

            print("Temporary file and directory will be removed by context manager.")

        return jsonify(
            {
                "language": info.language,
                "language_probability": info.language_probability,
                "transcription": full_transcription,
            }
        )

    except Exception as e:
        print(f"Error during transcription: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
