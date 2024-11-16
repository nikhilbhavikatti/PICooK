import os
import shutil
import glob
import re
import random


class DatasetGenerator():

    def __init__(self, mapping, data_dir="data"):
        self.mapping = mapping
        self.ingredient_images = glob.glob(os.path.join(data_dir, "ingredients", "*.jpg"))
        self.dish_images = glob.glob(os.path.join(data_dir, "dishes", "*.jpg"))
        self.dataset_dir = os.path.join(data_dir, "dish_ingredients")
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

    def generate(self):
        i = 0
        for dish in self.mapping:
            dish_name = dish["dish_name"]
            ingredients = dish["ingredients"]
            dish_img = list(filter(lambda x: re.search(f"^.*/{dish_name}_[0-9]+.jpg$", x), self.dish_images))
            ingredient_img = [list(filter(lambda x: re.search(f"^.*/{ingredient}_[0-9]+.jpg$", x), self.ingredient_images)) for ingredient in ingredients]
            
            if dish_img and all(ingredient_img):
                for img in dish_img:
                    folder_dir = os.path.join(self.dataset_dir, str(i))
                    os.mkdir(folder_dir)
                    shutil.copyfile(img, os.path.join(folder_dir, f"dish.jpg"))
                    for j, img in enumerate(ingredient_img):
                        shutil.copyfile(random.choice(img), os.path.join(folder_dir, f"ingredient{j}.jpg"))
                    i += 1
            else:
                print(f"Missing images (either dish or ingredient images) for {dish_name}")
            
            i += 1
