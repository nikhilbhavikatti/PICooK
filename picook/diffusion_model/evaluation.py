import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from einops import rearrange
from dataset import IngredientsDishDataset
from transformers import CLIPImageProcessor, CLIPVisionModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from diffusion_pipeline import PicookDiffusionPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_dir = "/data/work/jan/picook/picook/data/dishes_dataset"
model_name = "CompVis/stable-diffusion-v1-4"
output_dir = "./"
bs = 64
num_workers = 4
seed = 1337
weight_dtype =  torch.float32


test_dataset = IngredientsDishDataset(data_dir, is_test=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=bs,
    num_workers=num_workers,
)

image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=weight_dtype).to(device)
pipeline = PicookDiffusionPipeline.from_pretrained(model_name, image_encoder=image_model, torch_dtype=weight_dtype).to(device)
pipeline.set_progress_bar_config(disable=False)

# load attention processors
pipeline.load_lora_weights(output_dir)


def plot_images():
    # run inference
    # Get the image embedding for conditioning

    with torch.no_grad():
        batch = next(iter(test_dataloader))
        batch_size = batch["pixel_values"].shape[0]

        inp = batch["preprocessed_ingredient_images"].to(device, dtype=weight_dtype)
        inp = rearrange(inp, "b n c h w -> (b n) c h w")
        outputs = image_model(inp)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        encoder_hidden_states = rearrange(pooled_output, "(b n) c -> b n c", b=batch_size)

        # attention mask for unet
        attention_mask = (torch.arange(20)[None, :].repeat(batch_size, 1) < batch["len"][:, None]).int()
        attention_mask = attention_mask.to(device)

        # run pipeline
        generator = torch.Generator(device=device)
        generator = generator.manual_seed(42)
        images = pipeline(encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                batch_size=batch_size,
                                num_inference_steps=30,
                                guidance_scale=1.0,
                                generator=generator, is_training=False, output_type="numpy").images


        print(images.shape)
        # Create a figure with specified dimensions
        #fig, axes = plt.subplots(1, 4, figsize=(24, 13))
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        # Plot each image in the respective subplot
        #mapping_good = [1, 2, 6, 7, 21, 31, 60, 62]
        mapping_bad = [0, 16, 11, 48]
        for i, ax in enumerate(axes.flat):
            idx = mapping_bad[i]
            #idx = i
            ax.imshow(images[idx, :, :, :])
            ax.axis('off')  # Hide the axis

        # Adjust layout
        plt.tight_layout()
        plt.savefig("first_batch_generated.pdf")


def compute_fid():
    # Run inference for FID score
    fid = FrechetInceptionDistance(feature=64)
    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            #pipeline = PicookDiffusionPipeline.from_pretrained(model_name, image_encoder=image_model, torch_dtype=weight_dtype).to(device)
            #pipeline.set_progress_bar_config(disable=False)

            batch_size = batch["pixel_values"].shape[0]

            inp = batch["preprocessed_ingredient_images"].to(device, dtype=weight_dtype)
            inp = rearrange(inp, "b n c h w -> (b n) c h w")
            outputs = image_model(inp)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            encoder_hidden_states = rearrange(pooled_output, "(b n) c -> b n c", b=batch_size)

            # attention mask for unet
            attention_mask = (torch.arange(20)[None, :].repeat(batch_size, 1) < batch["len"][:, None]).int()
            attention_mask = attention_mask.to(device)

            # run pipeline
            images = pipeline(encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                batch_size=batch_size,
                                num_inference_steps=30,
                                generator=generator, is_training=False, output_type="numpy").images

            # Get real images (denormalize)
            dishes = batch["pixel_values"]
            dishes = (dishes * 0.5) + 0.5
            fid.update(dishes.to(torch.uint8), real=True)

            fid.update(torch.tensor(images).to(torch.uint8).permute(0, 3, 1, 2), real=False)         
            
            #del pipeline    
            torch.cuda.empty_cache()


    fid = fid.compute()
    print(f"FID Score:", fid)


plot_images()
#compute_fid()
