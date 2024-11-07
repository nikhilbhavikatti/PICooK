from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import os


class ImageValidator():

    def __init__(self, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                               device_map=self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                       device_map=self.device)
    
    def __get_images(self, path):
        images = []
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                images.append(Image.open(os.path.join(path, file)))
        return images
    
    def validate_image(images, class_label, batch_size=32):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = [Image.open(requests.get(url, stream=True).raw)] * 32

        inputs = self.processor(text=["cat", "dog", "food"], images=image, return_tensors="pt", padding=True).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        print(probs)
    