import glob
import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor


transform = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


def preprocess(path):
    folders = glob.glob(os.path.join(path, "*/"))
    max_seq_len = 20

    dishes = []
    ingredients = []
    len = []
    for folder in tqdm(folders):
        ingredients_ = glob.glob(os.path.join(folder, "ingredient*.jpg"))
        dish = glob.glob(os.path.join(folder, "dish.jpg"))
        
        try:
            ingredients_img = image_processor(images=[np.array(Image.open(img).convert("RGB")) for img in ingredients_], return_tensors="pt")["pixel_values"]
            length = ingredients_img.shape[0]
            padding = torch.zeros(max_seq_len - length, 3, 224, 224)
            ingredients.append(torch.concat((ingredients_img, padding), dim=0)[None, ...])
            len.append(length)

            dishes.append(transform(Image.open(dish[0]).convert("RGB"))[None, ...])
        except Exception as e:
            print(e)

    ingredients = torch.concat(ingredients, dim=0)
    dishes = torch.concat(dishes, dim=0)
    len = torch.tensor(len)

    print(ingredients.shape, dishes.shape, len.shape)

    torch.save(ingredients, os.path.join(path, "ingredients.pt"))
    torch.save(len, os.path.join(path, "len.pt"))
    torch.save(dishes, os.path.join(path, "dishes.pt"))


preprocess(path = "/data/work/ac141923/picook/picook/data/dishes_dataset/train")
preprocess(path = "/data/work/ac141923/picook/picook/data/dishes_dataset/test")