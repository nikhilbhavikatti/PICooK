from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map=device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device_map=device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = [Image.open(requests.get(url, stream=True).raw)]
inputs = processor(text=["cat", "dog", "food"], images=image, return_tensors="pt", padding=True).to(device)

print(inputs)

image_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", device_map=device)
outputs = image_model(inputs["pixel_values"]).image_embeds
print(outputs.shape)
#torch.Size([1, 768])

#print(inputs.shape)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
#print(probs)