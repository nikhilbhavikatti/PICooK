import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from dataset import IngredientsDishDataset
from diffusers import StableDiffusionPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths and model settings
data_dir = "dishes_dataset"
output_dir_stable = "./stable_diffusion_outputs"
os.makedirs(output_dir_stable, exist_ok=True)
bs = 4
num_workers = 1
seed = 1337
weight_dtype = torch.float32

# Load datasets
test_dataset = IngredientsDishDataset(data_dir, is_test=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=bs,
    num_workers=num_workers,
)

# Load Stable Diffusion pipeline
pipeline_stable = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
).to(device)
pipeline_stable.enable_attention_slicing()

# Generate images using Stable Diffusion
def generate_images_stable(dish_mapping):
    generated_images = []

    for entry in tqdm(dish_mapping["data"], desc="Generating images with Stable Diffusion"):
        # Ingredient prompt
        ingredient_list = ", ".join(entry["ingredients"])
        prompt_ingredients = f"Generate an image of a dish with {ingredient_list}."
        image_ingredients = pipeline_stable(prompt_ingredients, num_inference_steps=30).images[0]
        generated_images.append((entry["dish_name"], "ingredients", image_ingredients))

        # Dish name prompt
        prompt_dish_name = f"Generate an image of the dish {entry['dish_name']}."
        image_dish = pipeline_stable(prompt_dish_name, num_inference_steps=30).images[0]
        generated_images.append((entry["dish_name"], "dish_name", image_dish))

        # Save images
        image_ingredients.save(f"{output_dir_stable}/{entry['dish_name']}_ingredients.png")
        image_dish.save(f"{output_dir_stable}/{entry['dish_name']}_dish_name.png")

    return generated_images

def load_images_from_folder(folder_path, image_size=(224, 224), file_format="png"):

    image_files = [f for f in os.listdir(folder_path) if f.endswith(file_format)]
    images = []

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Lambda(lambda x: (x * 255).clamp(0, 255).byte())  # Scale to [0, 255] and convert to uint8
    ])

    for file in image_files:
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path).convert("RGB")
        images.append(transform(image))

    return torch.stack(images)  # Returns a tensor of shape (N, 3, H, W) with dtype=torch.uint8

def compute_fid_score(real_images, generated_images):

    fid = FrechetInceptionDistance(feature=64).to(real_images.device)

    # Update FID metric with real and generated images
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    return fid.compute().item()

if __name__ == "__main__":
    # Load dish-ingredient mapping
    with open("dish_ingredient_mapping_inverse_all.json", "r") as file:
        dish_mapping = json.load(file)

    # Generate images with Stable Diffusion
    images_stable = generate_images_stable(dish_mapping)

    # Paths to folders
    folder_generated = "./stable_diffusion_outputs_ing"  # Folder containing generated images (PNG format)
    folder_real = "./actual_dishes"       # Folder containing real images (JPG format)

    # Load images from folders
    generated_images = load_images_from_folder(folder_generated, image_size=(224, 224), file_format="png")
    real_images = load_images_from_folder(folder_real, image_size=(224, 224), file_format="jpg")

    # Ensure images are on the same device (CPU or GPU)
    generated_images = generated_images.to(device)
    real_images = real_images.to(device)

    # Compute and print FID score
    print("Computing FID....")
    fid_score = compute_fid_score(real_images, generated_images)
    print(f"FID Score: {fid_score}")

# FID score for Prompt with ingredient list : 2.23
# FID score for Prompt with dish name       : 1.86