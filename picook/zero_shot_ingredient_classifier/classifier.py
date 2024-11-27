from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os
import re
from tqdm import tqdm
import shutil
import json

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

        # create a path for corrupted images and move them there
        corrupted_path = os.path.join(folder_path, "corrupted_images")
        if not os.path.exists(corrupted_path):
            os.mkdir(corrupted_path)

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
                shutil.move(path, os.path.join(corrupted_path, file))
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

    def evaluate_images(self, dataset_path, labels_path, wrong_path=None, move_wrong_images=False):
        # Load ground-truth labels
        with open(labels_path, 'r') as f:
            ground_truth = json.load(f)

        images, labels, file_names = self.__get_images(dataset_path)

        tp, tn, fp, fn = 0, 0, 0, 0
        total = len(images)

        # Ensure wrong_path exists if move_wrong_images is True
        if move_wrong_images and wrong_path and not os.path.exists(wrong_path):
            os.mkdir(wrong_path)

        for i in tqdm(range(0, total, self.batch_size)):
            image_batch = images[i:i + self.batch_size]
            labels_batch = labels[i:i + self.batch_size]
            file_name_batch = file_names[i:i + self.batch_size]

            unique_labels = list(set(labels_batch))
            unique_labels += [
                "animal", "person", "house", "vehicle", "landscape"
            ]
            text = [f"a photo of a {label}" for label in unique_labels]

            with torch.no_grad():
                inputs = self.processor(text=text, images=image_batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()
                pred_classes = np.argpartition(probs, -self.top_k, axis=1)[:, -self.top_k:]

            for j in range(len(image_batch)):
                file_name = file_name_batch[j]
                true_label = ground_truth[file_name]["label"]
                is_correct = ground_truth[file_name]["is_correct"]

                predicted_labels = [unique_labels[k] for k in pred_classes[j]]
                #print(true_label, predicted_labels, is_correct)

                if not is_correct:  # Intentionally wrong image
                    if true_label not in predicted_labels:  # Correctly classified as wrong
                        tn += 1
                        if move_wrong_images and wrong_path:
                            shutil.move(
                                os.path.join(dataset_path, file_name),
                                os.path.join(wrong_path, file_name)
                            )
                    else:  # Incorrectly classified as correct
                        fp += 1
                else:  # Correct image
                    if true_label in predicted_labels:  # Correctly classified as correct
                        tp += 1
                    else:  # Incorrectly classified as wrong
                        fn += 1
                        if move_wrong_images and wrong_path:
                            shutil.move(
                                os.path.join(dataset_path, file_name),
                                os.path.join(wrong_path, file_name)
                            )

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(tp, tn, fp, fn)
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_images": total
        }
        return metrics
