import torch
from transformers import CLIPProcessor, CLIPModel
import transformers
from huggingface_hub import login
import json
import re

#adjust path
import ingredients_dishes
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import shutil

login(token='token')

class IngredientClassifier():
    """
    Classifies images of ingredients using CLIP model.
    params:
        ingredient_list: list of ingredients to classify
        batch_size: batch size for inference
        confidence: confidence for classification (how confidence the model should be to classify the image). If not confident enough
                    the image will be classified as "Unknown"
    """
    def __init__(self, ingredient_list, batch_size=64, confidence=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                               device_map=self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                       device_map=self.device)
        self.ingredient_list = ingredient_list
        self.batch_size = batch_size
        self.confidence = confidence

    def __get_images(self, paths):
        images = []

        for path in paths:
            try:
                image = Image.open(path)
                images.append(image)
            except:
                print(f"Could not open image")
        return images

    def predict(self, path):
        images = self.__get_images(path)
        classes = []

        total = len(images)
        for i in tqdm(range(0, len(images), self.batch_size)):
            # prepare input
            image_batch = images[i:i+self.batch_size]

            unique_labels = list(set(self.ingredient_list))
            # add some wrong labels to find wrong images
            unique_labels += ["animal", "person", "house", "car", "landscape", "electronic device", "nothing to eat", "cartoon", "drawing"]
            num_labels = len(unique_labels)
            text = [f"a photo of a {label}" for label in unique_labels]

            # inference
            with torch.no_grad():
                inputs = self.processor(text=text, images=image_batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()
                pred_classes = np.argmax(probs, axis=1)

                for i, pred in enumerate(pred_classes):
                    if probs[i, pred] >= self.confidence:
                        classes.append(unique_labels[pred])
                    else:
                        classes.append("Unknown")

        return classes

class RecipeGenerator():
    def __init__(self, ingredients, model_id="meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16):
        self.model_id = model_id

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map="auto")

    def generate_recipe(self, available_ingredients, max_length=256):

        prompt = f"I have these ingredients at home: {available_ingredients}. What could be a possible dish to cook and a recipie?"

        # Set context for LLM
        messages = [{"role": "system", "content": "You are a chef who outputs a dish from given ingredients and a recipie if asked. You only respond with the dish followed by a simicolon followed by a list representing the recipie. If there are missing ingredients for your dish you can add them. You do not respond with any additional text."},
                    {"role": "user", "content": prompt}]
        
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        out = self.pipeline(
            messages,
            max_new_tokens=max_length,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )[0]["generated_text"][-1]["content"]

        return out
        
ingredients_flat_list = []

for category, items in ingredients_dishes.ingredients.items():
    ingredients_flat_list.extend(items)

print(ingredients_flat_list)

ingredient_classifier = IngredientClassifier(ingredients_flat_list, batch_size=32, confidence=0.3)
recipe_generator = RecipeGenerator(ingredients_flat_list)


folder_path = "./2296"  # Replace with the name of your folder
image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
print(image_paths)

predictions = ingredient_classifier.predict(image_paths)

print(predictions)

filtered_predictions = [prediction for prediction in predictions if prediction != "Unknown"]

print(filtered_predictions)

recipe = recipe_generator.generate_recipe(filtered_predictions)
print(recipe)

"""
213
['Unknown', 'paprika', 'venison', 'chili pepper', 'parsnip', 'scallion', 'garlic', 'soy sauce', 'deli meats', 'peanut oil', 'chicken']
['paprika', 'venison', 'chili pepper', 'parsnip', 'scallion', 'garlic', 'soy sauce', 'deli meats', 'peanut oil', 'chicken']
Pan-Seared Venison with Spicy Parsnip Relish;
1. Preheat oven to 400°F (200°C).
2. Peel and chop 2 parsnips into small cubes.
3. Boil the parsnips in water for 10-12 minutes until tender.
4. In a pan, heat 2 tablespoons of peanut oil over medium heat.
5. Add 1 minced garlic and 1 minced chili pepper; sauté for 1 minute.
6. Add 1 cup of chopped scallions and cook for another minute.
7. Add 1 cup of boiled parsnips and stir to combine.
8. Season with 1 teaspoon of paprika and 1 tablespoon of soy sauce.
9. Meanwhile, season the venison with salt and pepper; pan-fry in 1 tablespoon of peanut oil until browned.
10. Serve the venison with the spicy parsnip relish.

1328
['black pepper', 'hot sauce', 'allspice', 'garlic', 'soy sauce', 'shallot', 'chicken', 'butter']
['black pepper', 'hot sauce', 'allspice', 'garlic', 'soy sauce', 'shallot', 'chicken', 'butter']
Asian-Style Chicken with Garlic Butter; 
- 1 chicken breast
- 2 cloves of garlic, minced
- 2 tablespoons of butter
- 1 teaspoon of allspice
- 1/2 teaspoon of black pepper
- 1/4 teaspoon of hot sauce
- 2 tablespoons of soy sauce

1805
['flaxseed', 'garlic', 'buttermilk', 'lime', 'Unknown', 'chili pepper', 'shrimp', 'cumin', 'Unknown', 'catfish', 'shrimp', 'fennel', 'olive oil']
['flaxseed', 'garlic', 'buttermilk', 'lime', 'chili pepper', 'shrimp', 'cumin', 'catfish', 'shrimp', 'fennel', 'olive oil']
Shrimp and Catfish Tacos; 
- 1 cup catfish, cut into small pieces
- 1/2 cup shrimp, peeled and deveined
- 2 cloves garlic, minced
- 1/2 teaspoon cumin
- 1/4 teaspoon chili pepper
- 1/4 teaspoon lime juice
- 2 tablespoons olive oil
- 1/2 cup fennel, sliced
- 2 tablespoons buttermilk
- Lime wedges for serving

2270
['cardamom', 'turmeric', 'shallot', 'green peas', 'cilantro', 'turmeric', 'cumin', 'cinnamon', 'chicken', 'ghee', 'black pepper', 'garlic']
['cardamom', 'turmeric', 'shallot', 'green peas', 'cilantro', 'turmeric', 'cumin', 'cinnamon', 'chicken', 'ghee', 'black pepper', 'garlic']
Chicken Tikka Masala;
1. 1 1/2 pounds boneless, skinless chicken breast or thighs, cut into 1 1/2-inch pieces
2. 2 medium shallots, minced
3. 2 cloves garlic, minced
4. 1-inch piece of ginger, grated
5. 1 teaspoon ground cumin
6. 1 teaspoon ground coriander
7. 1/2 teaspoon ground cinnamon
8. 1/2 teaspoon ground cardamom
9. 1/2 teaspoon ground turmeric
10. 1/2 teaspoon black pepper
11. 1/4 cup ghee
12. 1 cup chicken broth
13. 1 cup heavy cream
14. 1/2 cup chopped cilantro
15. Salt, to taste

2296
['olive oil', 'tomato', 'shallot', 'garlic', 'wheat', 'rice', 'lemon', 'bulgur', 'coriander leaves']
['olive oil', 'tomato', 'shallot', 'garlic', 'wheat', 'rice', 'lemon', 'bulgur', 'coriander leaves']
Tabbouleh; 
- 1 cup bulgur
- 1 cup chopped fresh coriander leaves
- 1 cup chopped fresh parsley
- 2 cloves garlic, minced
- 1/2 cup chopped fresh tomato
- 1/4 cup olive oil
- 2 tablespoons lemon juice
- Salt to taste
"""