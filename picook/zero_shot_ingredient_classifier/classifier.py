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


class IngredientClassifier():
    """
    Classifies images of ingredients using CLIP model.
    params:
        ingredient_list: list of ingredients to classify
        batch_size: batch size for inference
        confidence: confidence for classification (how confidence the model should be to classify the image). If not confident enough
                    the image will be classified as "Unknown"
    """
    def __init__(self, ingredient_list, batch_size=64, confidence=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                               device_map=self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                       device_map=self.device)
        self.ingredient_list = ingredient_list
        self.batch_size = batch_size
        self.confidence = confidence

    def __get_images(self, paths):
        images = []

        for path in paths:
            try:
                image = Image.open(path)
                images.append(image)
            except:
                print(f"Could not open image {file}")
        return images
    
    def predict(self, path):
        images = self.__get_images(path)
        classes = []

        total = len(images)
        for i in tqdm(range(0, len(images), self.batch_size)):
            # prepare input
            image_batch = images[i:i+self.batch_size]

            unique_labels = list(set(self.ingredient_list))
            # add some wrong labels to find wrong images
            unique_labels += ["animal", "person", "house", "car", "landscape", "electronic device", "nothing to eat", "cartoon", "drawing"]
            num_labels = len(unique_labels)
            text = [f"a photo of a {label}" for label in unique_labels]

            # inference
            with torch.no_grad():
                inputs = self.processor(text=text, images=image_batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()
                pred_classes = np.argmax(probs, axis=1)

                for i, pred in enumerate(pred_classes):
                    if probs[i, pred] >= self.confidence:
                        classes.append(unique_labels[pred])
                    else:
                        classes.append("Unknown")
            
        return classes


if __name__ == "__main__":
    import sys
    sys.path.append("../../")
    from picook.config.ingredients_dishes import ingredients
    ingredient_list = [ingredient for list_of_ingredients in ingredients.values() for ingredient in list_of_ingredients]
    classifier = IngredientClassifier(ingredient_list, confidence=0.5)
    classes = classifier.predict(["../../data/ingredients/alcohol_0.jpg", "../../data/ingredients/avocado_1.jpg"])
    print(classes)


## Case 1 top_k = 5
#   Accuracy : 0.6715
#   Precision : 0.6036
#   Recall : 1.0
#   F1-score : 0.7528

## Case 2 top_k = 3
#   Accuracy : 0.7842
#   Precision : 0.6987
#   Recall : 0.9992
#   F1-score : 0.8224

## Case 3 top_k = 1
#   Accuracy : 0.9055
#   Precision : 0.8465
#   Recall : 0.9905
#   F1-score : 0.9129


### Results on dish dataset - Total images : 8000
## Case 1 top_k = 5
#   Accuracy : 0.6863
#   Precision : 0.6145
#   Recall : 0.9997
#   F1-score : 0.7612

## Case 2 top_k = 3
#   Accuracy : 0.8120
#   Precision : 0.7269
#   Recall : 0.9995
#   F1-score : 0.8416

## Case 3 top_k = 1
#   Accuracy : 0.9256
#   Precision : 0.8744
#   Recall : 0.9940
#   F1-score : 0.9303