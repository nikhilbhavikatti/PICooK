import torch
from einops import rearrange
from dataset import IngredientsDishDataset
from transformers import CLIPImageProcessor, CLIPVisionModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from functools import partial

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score

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
    mean_clip_score = 0
    good_images = 0

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def calculate_clip_score(images, prompts):
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)


    with torch.no_grad():
        for batch in tqdm(test_dataloader):
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
            dishes = ((dishes * 0.5) + 0.5) * 255
            dishes = dishes.clamp(0, 255).to(torch.uint8)
            fid.update(dishes, real=True)

            images = (torch.tensor(images) * 255).clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2)
            fid.update(images, real=False)


            # clip score
            score = clip_score_fn(images, ["a photo of something to eat like food or a cooked dish"] * batch_size).detach()
            mean_clip_score += score * batch_size
            if score > 0.75:
                good_images += 1 * batch_size
            
            # clear memory  
            torch.cuda.empty_cache()



    fid = fid.compute()
    print(f"FID Score:", fid)

    good_fraction = good_images / len(test_dataset)
    print(f"Good images fraction:", good_fraction)

    mean_clip_score = mean_clip_score / len(test_dataset)
    print(f"Mean CLIP Score:", mean_clip_score)


#plot_images()
compute_fid()
