import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ConvNextFeatureExtractor

# Define the Flask app
app = Flask(__name__)

# Load the model and feature extractor
MODEL_PATH = "model"

try:
    # Load model and feature extractor
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    feature_extractor = ConvNextFeatureExtractor.from_pretrained(MODEL_PATH)
    print("Model and feature extractor loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model or feature extractor: {e}")

# Define the classify_image function
def classify_image(image: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_idx = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class_idx].item()

        confidence_threshold = 0.7
        if confidence < confidence_threshold:
            return {
                "prediction": "The image does not appear to be related to any known disease.",
                "confidence": confidence,
            }

        return {
            "prediction": model.config.id2label[predicted_class_idx],
            "confidence": confidence,
        }
    except Exception as e:
        raise RuntimeError(f"Error during image classification: {e}")

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    try:
        image_bytes = file.read()
        result = classify_image(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
