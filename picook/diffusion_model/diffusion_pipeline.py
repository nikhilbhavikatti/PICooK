import torch
from typing import Optional, Tuple, Union
from diffusers import ImagePipelineOutput
from diffusers import DiffusionPipeline
from einops import rearrange


class PicookDiffusionPipeline(DiffusionPipeline):
    """
    Diffusion pipeline for picook.
    """

    def __init__(self, vae, unet, scheduler, image_encoder):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler, image_encoder=image_encoder)
        
    @torch.no_grad()
    def __call__(
        self,
        images: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            images (`torch.Tensor`, *optional*, defaults to `None`):
                The images to condition the model on. If it is provided, `encoder_hidden_states` must be `None`.
            encoder_hidden_states (`torch.Tensor`, *optional*, defaults to `None`):
                The hidden states of the encoder model to condition the diffusion model. If it is provided, `images` must be `None`.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        # Prepare conditioning factors
        if images is None and encoder_hidden_states is None:
             raise ValueError("`images` or `encoder_hidden_states` must be provided.")
        elif images is not None and encoder_hidden_states is not None:
            raise ValueError("Only one of `images` or `encoder_hidden_states` can be provided.")
        elif images is not None and encoder_hidden_states is None:
            images = rearrange(images, "b n c h w -> (b n) c h w")
            outputs = self.image_model(images)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            encoder_hidden_states = rearrange(pooled_output, "(b n) c -> b n c", b=batch_size)
        
        # Sample gaussian noise to begin loop
        latent = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            device=self.device,
            generator=generator)
        latent = latent.to(self.device)

        # scale
        latent = latent * self.scheduler.init_noise_sigma

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            latent = self.scheduler.scale_model_input(latent, timestep=t)

            # 1. predict noise model_output
            model_output = self.unet(latent, t, encoder_hidden_states=encoder_hidden_states).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            latent = self.scheduler.step(model_output, t, latent).prev_sample
        
        # decode
        latent = (1 / self.vae.config.scaling_factor) * latent
        image = self.vae.decode(latent.to(torch.bfloat16)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    