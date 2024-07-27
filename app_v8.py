# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5 models."""

import argparse
import io
import os
import json
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import easyocr
import numpy as np
import cv2

lightstack = Flask(__name__)
models = {}
readers = {}


DETECTION_URL = "/v1/object-detection/<model_name>"


def get_ocr(image, model_name):
    """Perform OCR on an image using the specified model name."""
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image_hist = cv2.equalizeHist(image_gray)
    _, image_bin_thresh = cv2.threshold(image_hist, 120, 255, cv2.THRESH_BINARY) 
    result = readers[model_name].readtext(image_bin_thresh)
    return result


@lightstack.route(DETECTION_URL, methods=["POST"])
def predict(model_name):
    """Predict and return object detections in JSON format given an image and model name via a Flask REST API POST
    request.
    """
    if request.method != "POST":
        return jsonify({"success": False, "message": "Only POST method is supported"})

    if request.files.get("image"):
        # Read the image from the request
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        # Check if the model is loaded
        if model_name in models:
            # Perform prediction
            results = models[model_name].predict(
                im, imgsz=(640, 640), conf=0.5
            )  # reduce size=320 for faster inference

            # Convert results to JSON
            records = json.loads(results[0].tojson())
            predictions = (
                [
                    {
                        "label": r["name"],
                        "confidence": r["confidence"],
                        "x_min": r["box"]["x1"],
                        "y_min": r["box"]["y1"],
                        "x_max": r["box"]["x2"],
                        "y_max": r["box"]["y2"],
                    }
                    for r in records
                ]
                if len(records) > 0
                else []
            )

            # Perform OCR on the detected objects
            for i, pred in enumerate(predictions):
                if pred["label"] == "plate":
                    x_min, y_min, x_max, y_max = (
                        pred["x_min"],
                        pred["y_min"],
                        pred["x_max"],
                        pred["y_max"],
                    )
                    cropped_im = im.crop((x_min, y_min, x_max, y_max))
                    text = get_ocr(cropped_im, model_name)
                    print(text)
                    # predictions[i]["text"] = str(text)

            return jsonify(
                {
                    "success": True if len(records) > 0 else False,
                    "predictions": predictions,
                    "duration": results[0].speed,  # Optionally calculate duration
                }
            )
        else:
            return jsonify(
                {
                    "success": False,
                    "predictions": [],
                    "message": f"Model {model_name} not found",
                }
            )

    return jsonify(
        {"success": False, "predictions": [], "message": "Image file not provided"}
    )


def load_models(models_dir):
    """Load all YOLOv5 models from a given directory."""
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".engine")]
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[
            0
        ]  # Use filename without extension as model name
        print(f"Loading model: {model_name} from {model_path}")
        # Load the model using YOLOv8
        models[model_name] = YOLO(model_path, task="detect")

        print("Loading OCR model")
        readers[model_name] = easyocr.Reader(
            ["en"],
            model_storage_directory="/app/models/easyocr/",
            download_enabled=False,
        )


def initialize_app():
    """Initialize the app, loading models and any other setup tasks."""
    models_dir = os.environ.get("MODELS_DIR", "/app/models/")
    load_models(models_dir)


# Call initialization
initialize_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument(
        "--models-dir",
        default="/app/models/",
        help="directory containing model files",
    )
    opt = parser.parse_args()

    # Set the models directory environment variable
    os.environ["MODELS_DIR"] = opt.models_dir

    # Load all models from the specified directory
    initialize_app()

    lightstack.run(host="0.0.0.0", port=opt.port)
