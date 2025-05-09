import os
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS # New third-party import
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
        # Ensure the temporary directory is specific and cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a more robust way to name the temp file, handling empty filenames better
            filename = audio_file.filename if audio_file.filename else "audio_upload"
            temp_audio_path = os.path.join(temp_dir, filename)
            audio_file.save(temp_audio_path)
            print(f"Audio file saved to temporary path: {temp_audio_path}")

            # Perform transcription
            # Ensure model is not None again, just in case
            if model is None:
                print("Model became None before transcription") # Should not happen if initial check works
                return jsonify({"error": "Model not available for transcription"}), 500

            segments, info = model.transcribe(temp_audio_path, word_timestamps=True)

            print(
                "Detected language '%s' with probability %f"
                % (info.language, info.language_probability)
            )

            transcription_text_parts = []
            for segment in segments:
                transcription_text_parts.append(
                    "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
                )
                # Example for word timestamps (can be added to response if needed)
                # for word in segment.words:
                #     print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

            full_transcription = "\n".join(transcription_text_parts)
            print(f"Transcription: {full_transcription}")

            # No need to manually remove temp_audio_path or temp_dir,
            # tempfile.TemporaryDirectory() handles cleanup on exit of `with` block.
            print("Temporary file and directory will be removed.")

        return jsonify(
            {
                "language": info.language,
                "language_probability": info.language_probability,
                "transcription": full_transcription,
            }
        )

    except Exception as e:
        # Log the full exception for debugging
        import traceback
        print(f"Error during transcription: {e}\n{traceback.format_exc()}")
        # No manual cleanup of temp_dir needed here if using `with` statement above
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    # For development, Flask's dev server is fine.
    # Debug mode should ideally be False for any kind of deployment or testing with gunicorn
    # Gunicorn will manage the workers and address binding.
    # When running with `gunicorn infer:app`, this __main__ block is not executed.
    # If you run `python infer.py` directly, this dev server starts.
    app.run(host="0.0.0.0", port=5000, debug=False)
