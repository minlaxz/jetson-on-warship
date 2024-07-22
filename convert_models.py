from ultralytics import YOLO
import os

def convert_models_to_engine(models_dir):
    for file_name in os.listdir(models_dir):
        if file_name.endswith('.pt'):
            model_path = os.path.join(models_dir, file_name)
            model = YOLO(model_path)
            engine_path = model_path.replace('.pt', '.engine')
            print(f"Converting {model_path} to {engine_path}")
            model.export(format='engine', device="cpu")

if __name__ == "__main__":
    models_directory = "/app/models"
    convert_models_to_engine(models_directory)
