import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained ResNet-50 model
model_path = "model/resnet50_dr_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.resnet50(pretrained=False)  # No pretraining, load our weights
model.fc = nn.Linear(model.fc.in_features, 5)  # Adjust for 5 classes
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()



# Define class labels
class_labels = ["Mild", "Moderate","No_DR", "Proliferate_DR", "Severe"]

# Image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

# Prediction function
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]

# Home Page (File Upload)
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            prediction = predict(filepath)
            return render_template("result.html", filename=filename, prediction=prediction)
    
    return render_template("index.html")

# Route to serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
