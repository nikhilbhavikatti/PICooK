from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map=device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map=device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = [Image.open(requests.get(url, stream=True).raw)] * 32

inputs = processor(text=["cat", "dog", "food"], images=image, return_tensors="pt", padding=True).to(device)

#print(inputs.shape)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)