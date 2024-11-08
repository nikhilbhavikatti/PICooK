#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

source /home/jan/miniforge3/etc/profile.d/conda.sh
conda activate fm24
cd /home/jan/git/picook/picook/diffusion_model

# food classification dataset
#accelerate launch train_images_to_image_lora.py   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"   --dataset_name="muhammadbilal5110/indian_food_images"   --dataloader_num_workers=1   --resolution=512 --center_crop --random_flip   --train_batch_size=16   --gradient_accumulation_steps=4   --max_train_steps=15000   --learning_rate=1e-04   --max_grad_norm=1   --lr_scheduler="cosine" --lr_warmup_steps=0   --output_dir=.  --checkpointing_steps=1000   --seed=1337  --report_to="wandb"

# food classification dataset
accelerate launch train_images_to_image_lora.py   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"   --imgs2img --data_dir="/home/jan/git/picook/data/dishes_dataset"  --dataloader_num_workers=1   --resolution=512 --center_crop --random_flip   --train_batch_size=16   --gradient_accumulation_steps=4   --max_train_steps=15000   --learning_rate=1e-04   --max_grad_norm=1   --lr_scheduler="cosine" --lr_warmup_steps=0   --output_dir=.  --checkpointing_steps=1000   --seed=1337  --report_to="wandb"