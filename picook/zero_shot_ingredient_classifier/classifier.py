from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os
import re
from tqdm import tqdm
import shutil

class ImageValidator():

    def __init__(self, batch_size=64, top_k=3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                               device_map=self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                       device_map=self.device)
        self.batch_size = batch_size
        self.top_k = top_k

    def __get_images(self, folder_path):
        images = []
        labels = []
        file_names = []
        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)
            if os.path.isdir(path):
                # skip directories
                continue

            try:
                image = Image.open(path)
                label = re.findall(r"(.*)_[0-9]+", file)[0]
                labels.append(label)
                images.append(image)
                file_names.append(file)
            except:
                print(f"Could not open image {file}")
        return images, labels, file_names
    
    def validate_images(self, path, wrong_path, move_wrong_images=False):
        # create the wrong dir if it does not exist
        if move_wrong_images and not os.path.exists(wrong_path):
            os.mkdir(wrong_path)

        images, labels, file_names = self.__get_images(path)

        wrong_files = []
        total = len(images)
        wrong_total = 0

        for i in tqdm(range(0, len(images), self.batch_size)):
            # prepare input
            image_batch = images[i:i+self.batch_size]
            labels_batch = labels[i:i+self.batch_size]
            file_name_batch = file_names[i:i+self.batch_size]

            unique_labels = list(set(labels_batch))
            # add some wrong labels to find wrong images
            unique_labels += ["animal", "person", "house", "car", "landscape", "electronic device", "nothing to eat", "cartoon", "drawing"]
            text = [f"a photo of a {label}" for label in unique_labels]

            num_labels_batch = np.array([unique_labels.index(label) for label in labels_batch])

            # inference
            with torch.no_grad():
                inputs = self.processor(text=text, images=image_batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()
                pred_classes = np.argpartition(probs, -self.top_k, axis=1)[:, -self.top_k:]
                
                for i in range(pred_classes.shape[0]):
                    if not np.isin(num_labels_batch[i], pred_classes[i]):
                        #print(f"Image {file_name} is not {label} but {unique_labels[pred]}")
                        if move_wrong_images:
                            shutil.move(os.path.join(path, file_name_batch[i]), os.path.join(wrong_path, file_name_batch[i]))
                        wrong_files.append(file_name_batch[i])
                        wrong_total += 1
            
        print(f"Wrong images: {wrong_total}/{total}")
        return wrong_files
