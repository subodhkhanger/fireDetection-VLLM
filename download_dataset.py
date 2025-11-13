from roboflow import Roboflow
import os
import shutil

# Your API key
API_KEY = "APIKEY"  # Replace with your actual key

# Initialize and download
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("middle-east-tech-university").project("fire-and-smoke-detection-hiwia")
dataset = project.version(2).download("coco")

# Move to expected location
target_dir = "./data/fire-detection/fire-and-smoke-detection-2"
os.makedirs(os.path.dirname(target_dir), exist_ok=True)

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
    
shutil.move(dataset.location, target_dir)
print(f"✓ Dataset ready at: {target_dir}")