import torch
from dataset import IngredientsDishDataset
import matplotlib.pyplot as plt
import numpy as np

test_dataset = IngredientsDishDataset("/data/work/ac141923/picook/picook/data/dishes_dataset", is_test=True)
test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=32,
            num_workers=0)
batch = next(iter(test_dataloader))
dishes = batch["pixel_values"].numpy()
dishes = (dishes * 0.5) + 0.5

# Create a figure with specified dimensions
fig, axes = plt.subplots(4, 8, figsize=(24, 12))

# Plot each image in the respective subplot
for i, ax in enumerate(axes.flat):
    ax.imshow(dishes[i].transpose((1,2,0)))
    ax.axis('off')  # Hide the axis

# Adjust layout
plt.tight_layout()
plt.savefig("first_batch.png")
