import os
import json
import re

# Directory containing the images
images_dir = "data/ingredients"  # Update with your image directory
json_file_path = "data/ingredients_labels.json"  # Path to save the JSON file

# Function to infer label from file name
def get_label(file_name):
    if "_w_i" in file_name:  # Indicates a wrong image
        label = file_name.split("_")[0]  # Extracts the label before `_`
        return label, False
    else:  # Correct image
        label = file_name.split("_")[0]  # Extracts the label before `_`
        return label, True

# Create JSON data
json_data = {}
for file_name in os.listdir(images_dir):
    if file_name.endswith((".jpg", ".jpeg", ".png")):  # Process only image files
        label, is_correct = get_label(file_name)
        json_data[file_name] = {
            "label": label,
            "is_correct": is_correct
        }

# Write to JSON file
with open(json_file_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"JSON file created at {json_file_path}")
