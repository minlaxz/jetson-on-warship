# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5s models."""

import argparse
import io
import os

import torch
from flask import Flask, request, jsonify
from PIL import Image

lightstack = Flask(__name__)
models = {}

DETECTION_URL = "/v1/object-detection/<model_name>"

@lightstack.route(DETECTION_URL, methods=["POST"])
def predict(model_name):
    """Predict and return object detections in JSON format given an image and model name via a Flask REST API POST
    request.
    """
    if request.method != "POST":
        return

    if request.files.get("image"):
        # Read the image from the request
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        # Check if the model is loaded
        if model_name in models:
            # Perform prediction
            results = models[model_name](
                im, size=640
            )  # reduce size=320 for faster inference
            
            # Convert results to JSON
            records = results.pandas().xyxy[0].to_dict(orient="records")
            predictions = [
                {
                    "label": r["name"],
                    "confidence": r["confidence"],
                    "x_min": r["xmin"],
                    "y_min": r["ymin"],
                    "x_max": r["xmax"],
                    "y_max": r["ymax"],
                }
                for r in records
            ]

            return jsonify({
                "success": True,
                "predictions": predictions,
                "duration": 0,  # Optionally calculate duration
            })

    return jsonify({"success": False, "predictions": []})

def load_models(models_dir):
    """Load all YOLOv5 models from a given directory."""
    model_files = [
        f for f in os.listdir(models_dir) if f.endswith('.engine')
    ]
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]  # Use filename without extension as model name
        print(f"Loading model: {model_name} from {model_path}")
        
        # Load the model using torch.hub
        models[model_name] = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            force_reload=True,
            skip_validation=True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument(
        "--models-dir",
        default="/app/models/",
        help="directory containing model files",
    )
    opt = parser.parse_args()

    # Load all models from the specified directory
    load_models(opt.models_dir)

    lightstack.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
