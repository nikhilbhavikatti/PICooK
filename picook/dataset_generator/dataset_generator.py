import os
import shutil
import glob
import re
import random
import numpy as np


class DatasetGenerator():

    def __init__(self, mapping, data_dir="data", train_test_split=0.8):
        self.mapping = mapping
        self.ingredient_images = glob.glob(os.path.join(data_dir, "ingredients", "*.jpg"))
        self.dish_images = glob.glob(os.path.join(data_dir, "dishes", "*.jpg"))
        self.dataset_dir = os.path.join(data_dir, "dish_ingredients")
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        
        self.train_test_split = train_test_split
        self.num_samples = 0

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
            
            #i += 1

        self.num_samples = i
    
    def split(self):
        # make folders for train and test
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        if not os.path.exists(self.train_dir): os.mkdir(self.train_dir)
        if not os.path.exists(self.test_dir): os.mkdir(self.test_dir)
        
        if self.num_samples == 0:
            self.num_samples = len(glob.glob(os.path.join(self.dataset_dir, "[0-9]*/")))
        
        idx = np.arange(self.num_samples)
        np.random.shuffle(idx)
        train_idx = idx[:int(self.num_samples * self.train_test_split)]
        test_idx = idx[int(self.num_samples * self.train_test_split):]
        print(f"Number of samples: {self.num_samples}, test samples: {len(test_idx)}, train samples: {len(train_idx)}")

        for folder_num in train_idx:
            folder_num = str(folder_num)
            folder_path = os.path.join(self.dataset_dir, folder_num)
            shutil.move(folder_path, os.path.join(self.train_dir, folder_num))
        
        for folder_num in test_idx:
            folder_num = str(folder_num)
            folder_path = os.path.join(self.dataset_dir, folder_num)
            shutil.move(folder_path, os.path.join(self.test_dir, folder_num))
