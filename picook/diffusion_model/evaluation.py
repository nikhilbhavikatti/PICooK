import torch
from einops import rearrange
from dataset import IngredientsDishDataset
from transformers import CLIPImageProcessor, CLIPVisionModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from functools import partial
import os
import glob
from PIL import Image
from functools import reduce

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.multimodal.clip_score import CLIPScore

from diffusion_pipeline import PicookDiffusionPipeline

from torchvision import transforms
from transformers import CLIPImageProcessor


y_transform = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

x_transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

resize =  transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512)
    ]
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_dir = "../../data/dishes_dataset"
model_name = "CompVis/stable-diffusion-v1-4"
output_dir = "./"
bs = 36
num_workers = 4
seed = 1337
weight_dtype =  torch.float32


test_dataset = IngredientsDishDataset(data_dir, is_test=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=True,
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
        generator = generator.manual_seed(69)
        images = pipeline(encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                batch_size=batch_size,
                                num_inference_steps=30,
                                guidance_scale=1.0,
                                generator=generator, is_training=False, output_type="numpy").images


        print(images.shape)
        # Create a figure with specified dimensions
        #fig, axes = plt.subplots(1, 4, figsize=(24, 13))
        fig, axes = plt.subplots(6, 6, figsize=(24, 24))

        # Plot each image in the respective subplot
        #mapping_good = [1, 2, 6, 7, 21, 31, 60, 62]
        #mapping_bad = [0, 16, 11, 48]
        for i, ax in enumerate(axes.flat):
            #idx = mapping_bad[i]
            idx = i
            ax.imshow(images[idx, :, :, :])
            ax.axis('off')  # Hide the axis

        # Adjust layout
        plt.tight_layout()
        plt.savefig("first_batch_generated_large_2.pdf")


def compute_fid():
    # Run inference for FID score
    fid = FrechetInceptionDistance(feature=64)
    mean_clip_score = 0
    good_images = 0

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")

    def calculate_clip_score(images, prompts):
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)

    clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")

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
                                is_training=False, output_type="numpy").images

            # Get real images (denormalize)
            dishes = batch["pixel_values"]
            dishes = ((dishes * 0.5) + 0.5) * 255
            dishes = dishes.clamp(0, 255).to(torch.uint8)
            fid.update(dishes, real=True)

            images = (torch.tensor(images) * 255).clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2)
            fid.update(images, real=False)


            # clip score
            score = clip_score_fn(dishes, ["a photo of a dish"] * batch_size).detach()
            print(score)
            mean_clip_score += score * batch_size
            if score > 0.75:
                good_images += 1 * batch_size
            
            clip_score_metric.update(dishes, ["a photo of a dish"] * batch_size)
            # clear memory  
            torch.cuda.empty_cache()



    fid = fid.compute()
    print(f"FID Score:", fid)

    num = clip_score_metric.compute()
    print(f"CLIP Score:", num)

    good_fraction = good_images / len(test_dataset)
    print(f"Good images fraction:", good_fraction)

    mean_clip_score = mean_clip_score / len(test_dataset)
    print(f"Mean CLIP Score:", mean_clip_score)


def plot_images_with_conditioning():
    # run inference
    # Get the image embedding for conditioning
    
    base_path = "../../data/dishes_dataset/test"
    #selection = [10, 1532, 1222, 2747] for paper
    selection = [213, 1328, 1805, 2270, 2296]
    max_sequence_length = 20
    img = []
    ing = []
    dishes = []

    for idx in selection:
        path = os.path.join(base_path, str(idx))
        ingredients = glob.glob(os.path.join(path, "ingredient*.jpg"))
        dish = glob.glob(os.path.join(path, "dish.jpg"))
        num_ingredients = torch.tensor([len(ingredients)])

        if len(ingredients) == 0 or len(dish) == 0:
            raise ValueError("No ingredients or dish found")

        ingredients = [Image.open(img).convert("RGB") for img in ingredients]
        ing.append([resize(ingredient_img) for ingredient_img in ingredients])
        ingredients = x_transform(images=ingredients, return_tensors="pt")["pixel_values"]
        padding = torch.zeros(max_sequence_length - num_ingredients, 3, 224, 224)
        ingredients = torch.concat((ingredients, padding), dim=0)
        dishes.append(resize(Image.open(dish[0]).convert("RGB")))

        with torch.no_grad():
            batch_size = 1

            inp = ingredients.unsqueeze(0).to(device, dtype=weight_dtype)
            inp = rearrange(inp, "b n c h w -> (b n) c h w")
            outputs = image_model(inp)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            encoder_hidden_states = rearrange(pooled_output, "(b n) c -> b n c", b=batch_size)

            # attention mask for unet
            attention_mask = (torch.arange(20)[None, :].repeat(batch_size, 1) < num_ingredients[:, None]).int()
            attention_mask = attention_mask.to(device)

            # run pipeline
            generator = torch.Generator(device=device)
            generator = generator.manual_seed(69)
            images = pipeline(encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                    batch_size=batch_size,
                                    num_inference_steps=30,
                                    guidance_scale=1.0,
                                    generator=generator, is_training=False, output_type="numpy").images
            images = (torch.tensor(images) * 255).clamp(0, 255).to(torch.uint8)
            img.append(images)
            im = Image.fromarray(images[0].cpu().numpy())
            im.save(f"generated_image_{idx}.jpeg")

    images = np.concatenate(img, axis=0)

    # Create a figure with specified dimensions
    #fig, axes = plt.subplots(1, 4, figsize=(24, 13))
    max_seq_len = reduce(lambda x, y: max(x, y), [len(x) for x in ing])
    fig, axes = plt.subplots(len(selection), max_seq_len+3, figsize=(26, 8), gridspec_kw={'width_ratios': [1] * (max_seq_len) + [0.3, 1, 1]})

    # Plot each image in the respective subplot
    for i in range(len(selection)):
        # ingredients
        for j, ingredient in enumerate(ing[i]):
            axes[i, j].imshow(ingredient)

        # pred
        axes[i, -2].imshow(images[i, :, :, :])

        # true
        axes[i, -1].imshow(dishes[i])
        
    # title
    axes[0,-2].set_title("Generated\nImage", fontsize=20)
    axes[0,-1].set_title("Ground Truth\nImage", fontsize=20)

    # hide all axis
    for ax in axes.flatten():
        ax.axis('off')  # Hide the axis

    # Adjust layout
    plt.tight_layout()
    plt.savefig("input_output.pdf")
    plt.savefig("input_output.png")



def plot_images_with_removed_conditioning():
    # run inference
    # Get the image embedding for conditioning
    
    base_path = "../../data/dishes_dataset/test"
    #selection = [10, 1532, 1222, 2747] for paper
    selection = [1222]
    max_sequence_length = 20
    img = []
    ing = []
    dishes = []

    for idx in selection:
        for j in range(0,8):
            path = os.path.join(base_path, str(idx))
            ingredients = glob.glob(os.path.join(path, "ingredient*.jpg"))
            if j > 0:
                ingredients = ingredients[:-j]
            dish = glob.glob(os.path.join(path, "dish.jpg"))
            num_ingredients = torch.tensor([len(ingredients)])

            if len(ingredients) == 0 or len(dish) == 0:
                raise ValueError("No ingredients or dish found")

            ingredients = [Image.open(img).convert("RGB") for img in ingredients]
            ing.append([resize(ingredient_img) for ingredient_img in ingredients])
            ingredients = x_transform(images=ingredients, return_tensors="pt")["pixel_values"]
            padding = torch.zeros(max_sequence_length - num_ingredients, 3, 224, 224)
            ingredients = torch.concat((ingredients, padding), dim=0)
            dishes.append(resize(Image.open(dish[0]).convert("RGB")))

            with torch.no_grad():
                batch_size = 1

                inp = ingredients.unsqueeze(0).to(device, dtype=weight_dtype)
                inp = rearrange(inp, "b n c h w -> (b n) c h w")
                outputs = image_model(inp)
                last_hidden_state = outputs.last_hidden_state
                pooled_output = outputs.pooler_output
                encoder_hidden_states = rearrange(pooled_output, "(b n) c -> b n c", b=batch_size)

                # attention mask for unet
                attention_mask = (torch.arange(20)[None, :].repeat(batch_size, 1) < num_ingredients[:, None]).int()
                attention_mask = attention_mask.to(device)
                print(attention_mask)

                # run pipeline
                generator = torch.Generator(device=device)
                generator = generator.manual_seed(1234)
                #generator = generator.manual_seed(5467)
                images = pipeline(encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
                                        batch_size=batch_size,
                                        num_inference_steps=30,
                                        guidance_scale=1.0,
                                        generator=generator, is_training=False, output_type="numpy").images
                images = (torch.tensor(images) * 255).clamp(0, 255).to(torch.uint8)
                img.append(images)
                im = Image.fromarray(images[0].cpu().numpy())
                im.save(f"generated_image_{idx}.jpeg")

    images = np.concatenate(img, axis=0)

    # Create a figure with specified dimensions
    #fig, axes = plt.subplots(1, 4, figsize=(24, 13))
    max_seq_len = reduce(lambda x, y: max(x, y), [len(x) for x in ing])
    fig, axes = plt.subplots(len(selection * 8), max_seq_len+2, figsize=(18, 20), gridspec_kw={'width_ratios': [1] * (max_seq_len) + [0.3, 1]})

    # Plot each image in the respective subplot
    for i in range(len(selection) * 8):
        # ingredients
        for j, ingredient in enumerate(ing[i]):
            axes[i, j].imshow(ingredient)

        # pred
        axes[i, -1].imshow(images[i, :, :, :])
        
    # title
    axes[0,-1].set_title("Generated\nImage", fontsize=20)

    # hide all axis
    for ax in axes.flatten():
        ax.axis('off')  # Hide the axis

    # Adjust layout
    plt.tight_layout()
    plt.savefig("input_output_cond_removed.pdf")
    plt.savefig("input_output_cond_removed.png")


#plot_images()
#compute_fid()
#plot_images_with_conditioning()
plot_images_with_removed_conditioning()