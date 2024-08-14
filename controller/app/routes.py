# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5 models."""

import argparse
import io
import os
import json
from PIL import Image
import easyocr
import numpy as np
import cv2
from ultralytics import YOLO

from flask import request, jsonify, make_response, render_template
from flask import current_app as lightstack

models = {}
readers = {}
records = {}
DETECTION_URL = "/api/v1/object-detection/<model_name>"


@lightstack.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html", records=[], host=request.host_url)


def get_ocr(image, model_name):
    """Perform OCR on an image using the specified model name."""
    image_np = np.array(image)
    image_zero = cv2.cvtColor(image_np, cv2.THRESH_TOZERO)
    result = readers[model_name].readtext(
        image_zero,
        detail=0,
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        width_ths=1.5,
        text_threshold=0.8,
    )
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
                    if len(text) >= 3:
                        _division, _plate, _model = text[:3]
                        others = text[3:]
                    else:
                        _division = text[0]
                        _plate = text[1]
                        _model = "Unknown"
                        others = text[2:]
                    predicted_text = f"Division:{_division} + Plate:{_plate} + Model:{_model}"
                    print(predicted_text)
                    print(others)
                    # for i in records["plates"]:
                    #     if i["plate"] == _plate:
                    #         print("Found record")
                    #     else:
                    #         print("Not found")
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


def load_models(yolo_models_dir):
    """Load all YOLOv5 models from a given directory."""
    yolo_model_files = [f for f in os.listdir(yolo_models_dir) if f.endswith(".engine")]
    for yolo_model_file in yolo_model_files:
        model_path = os.path.join(yolo_models_dir, yolo_model_file)
        model_name = os.path.splitext(yolo_model_file)[0]
        print(f"Loading model: {model_name} from {model_path}")
        models[model_name] = YOLO(model_path, task="detect")
        print("Loading OCR model")
        readers[model_name] = easyocr.Reader(
            ["en"],
            model_storage_directory="/home/app/ocr_models/",
            download_enabled=False,
        )
    with open("records.json", "r") as f:
        records["plates"] = json.load(f)
    print(f"Records loaded {json.dump(records)}")
    print("Models loaded")


def initialize_app():
    """Initialize the app, loading models and any other setup tasks."""
    yolo_models_dir = os.environ.get("YOLO_MODELS_DIR", "/home/app/yolo_models/")
    load_models(yolo_models_dir)


# Call initialization
initialize_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument(
        "--yolo-models-dir",
        default="/home/app/yolo_models/",
        help="directory containing YOLO model files",
    )
    opt = parser.parse_args()

    # Set the models directory environment variable
    os.environ["YOLO_MODELS_DIR"] = opt.yolo_models_dir

    # Load all models from the specified directory
    initialize_app()

    lightstack.run(host="0.0.0.0", port=opt.port)
