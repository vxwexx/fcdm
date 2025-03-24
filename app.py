import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image, ImageEnhance
import tensorflow.lite as tflite

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="CDM.tlite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess an image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    img = ImageEnhance.Contrast(img).enhance(2.0)  # Increase contrast
    img = img.resize(target_size)  # Resize to match model input size
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for model input
    return img_array

# Function to predict currency authenticity
def predict_currency(image_path, threshold=0.7):
    preprocessed_image = load_and_preprocess_image(image_path)
    
    # Run inference
    interpreter.set_tensor(input_details[0]["index"], preprocessed_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    return "Real Currency" if prediction >= threshold else "Fake Currency"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_currency(filepath)
        
        return render_template("index.html", prediction=result)

    return render_template("index.html")

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
