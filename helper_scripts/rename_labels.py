import os
import random
import shutil
from pathlib import Path

# Step 1: Define the dataset path and custom labels
imagenet_food_path = "wrng_imgs/"
output_path = "out_wrng_imgs/"
custom_labels = ['avocado', 'baking-powder', 'bay-leaves', 'beetroot', 'black-beans', 'black-pepper', 'brown-sugar' ,
                 'butter', 'Cabbage', 'Capsicum', 'carrots', 'Cauliflower', 'chicken-flesh', 'chili-powder', 'chopped-onion',
                 'cilantro-leaves', 'corn', 'Cucumber', 'cumin', 'diced-tomatoes', 'eggplant', 'eggs', 'flour', 'garlic',
                 'ginger', 'green-onions', 'ground-turmeric', 'honey', 'lemon', 'lettuce', 'oil', 'peas', 'potatoes',
                 'purple-onion', 'raddish', 'salt', 'soy-beans', 'spinach', 'sugar', 'turnip']

# Step 2: Create output directory
os.makedirs(output_path, exist_ok=True)

# Step 3: Gather all image file paths
image_paths = list(Path(imagenet_food_path).rglob("*.jpeg"))  # Update extension as needed
print(image_paths)
# Step 4: Shuffle images
random.shuffle(image_paths)  # Shuffle the image paths for randomness

# Step 5: Create counters for each label
label_counters = {label: 1 for label in custom_labels}

# Step 6: Rename images and save them with unique names
for image_path in image_paths:
    # Randomly pick a label
    label = random.choice(custom_labels)
    
    # Generate a unique filename for the label
    new_filename = f"{label}_{label_counters[label]:02d}_w_i.jpg"
    print(new_filename)
    label_counters[label] += 1  # Increment the counter for the label
    
    # Define the output path for the renamed image
    new_file_path = os.path.join(output_path, new_filename)
    
    # Copy and rename the image
    shutil.copy(image_path, new_file_path)

print("Images successfully renamed and labeled!")
