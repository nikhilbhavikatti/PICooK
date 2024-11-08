import glob
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class IngredientsDishDataset(Dataset):
    """
    Custom dataset for picook diffusion model.
    """

    def __init__(self, data_dir, y_transform, x_transform, is_test=False):
        #path = "/home/jan/git/picook/data/dishes_dataset/train"
        suffix = "test" if is_test else "train"
        folders = glob.glob(os.path.join(data_dir, suffix, "*/"))
        self.ingredients = []
        self.dishes = []
        for folder in folders:
            ingredients = glob.glob(os.path.join(folder, "ingredient*.jpg"))
            dish = glob.glob(os.path.join(folder, "dish.jpg"))
            self.ingredients.append(x_transform(images=[np.array(Image.open(img).convert("RGB")) for img in ingredients], return_tensors="pt")["pixel_values"])
            self.dishes.append(y_transform(Image.open(dish[0]).convert("RGB")))

        self.dishes = torch.stack(self.dishes, dim=0)
        self.dishes = self.dishes.to(memory_format=torch.contiguous_format).float()
    
    def __len__(self):
        return len(self.dishes)
    
    def __getitem__(self, idx):
        return { "pixel_values": self.dishes[idx], "preprocessed_ingredient_images": self.ingredients[idx] }
