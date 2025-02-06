import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModel

class IngredientsDishDataset_old(Dataset):
    """
    Custom dataset for picook diffusion model.
    """

    def __init__(self, data_dir, y_transform, x_transform, is_test=False, max_seq_length=20, use_cache=True):
        #path = "/home/jan/git/picook/data/dishes_dataset/train"
        suffix = "test" if is_test else "train"
        self.base_path = os.path.join(data_dir, suffix)
        self.dishes = []
        for folder in glob.glob(os.path.join(self.base_path, "*/")):
            ingredients = glob.glob(os.path.join(folder, "ingredient*.jpg"))
            dish = glob.glob(os.path.join(folder, "dish.jpg"))
            if len(ingredients) > 0 and len(dish) > 0:
                self.dishes.append(folder)

        self.x_transform = x_transform
        self.y_transform = y_transform
        self.max_sequence_length = max_seq_length

        self.cached_data = []
        self.use_cache = use_cache
    
    def __len__(self):
        return len(self.dishes)
    
    def __getitem__(self, idx):
        ingredients_paths = glob.glob(os.path.join(self.dishes[idx], "ingredient*.jpg"))
        dish_paths = glob.glob(os.path.join(self.dishes[idx], "dish.jpg"))

        images = [Image.open(img).convert("RGB") for img in ingredients_paths]
        padded_images = images + ([images[0]] * (self.max_sequence_length - len(images)))
        
        ingredients = self.x_transform(images=padded_images, return_tensors="pt")["pixel_values"]
        dishes = self.y_transform(Image.open(dish_paths[0]).convert("RGB"))

        return { "pixel_values": dishes, "preprocessed_ingredient_images": ingredients}

    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache


class IngredientsDishDataset(Dataset):
    """
    Custom dataset for picook diffusion model.
    """

    def __init__(self, data_dir, is_test=False):
        #path = "/home/jan/git/picook/data/dishes_dataset/train"
        suffix = "test" if is_test else "train"
        path = os.path.join(data_dir, suffix)

        self.ingredients = torch.load(os.path.join(path, "ingredients.pt"), weights_only=True)
        self.len = torch.load(os.path.join(path, "len.pt"), weights_only=True)
        self.dishes = torch.load(os.path.join(path, "dishes.pt"), weights_only=True)
    
    def __len__(self):
        return len(self.dishes)
    
    def __getitem__(self, idx):
        #return { "pixel_values": self.dishes[idx], "preprocessed_ingredient_images": self.ingredients[idx]}
        return { "pixel_values": self.dishes[idx], "preprocessed_ingredient_images": self.ingredients[idx], "len": self.len[idx]}



if __name__ == "__main__":
    # @janha Preprocessing our custom dataset.
    print("Loading imgs2img dataset")
    train_transforms = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224) if True else transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = IngredientsDishDataset("/data/work/ac141923/picook/picook/data/dishes_dataset", train_transforms, image_processor)


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=32,
        num_workers=2,
    )

    print(next(iter(train_dataloader)))
