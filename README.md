# :cake: PICooK: Photo Ingredients to Cooked Dish (Foundation Models Project 2024)

# Dataset Generation Pipeline

![Dataset generation pipeline](imgs/data_generation_pipeline.svg)

## Image Retrieval
```bash
export GOOGLE_API_KEY="<KEY>"
export GOOGLE_SEARCH_ENGINE_ID="<ID>"
python main.py --scrape
```

## Image Validation using Classifier
```bash
python main.py --validate
```

## Dish Generation
Generate token for LLaMa on HuggingFace, request permission for meta-llama/Llama-3.2-1B
If you want to use on your own device download it by using:
```bash
set HF_TOKEN=your_token_here
huggingface-cli download meta-llama/Llama-3.2-1B --include "original/*" --local-dir /path/to/save/directory/Llama-3.2-1B
```
However if wanting to use Google Colab, please insert your HuggingFace token in
```python
login(token='your_token-here')
```

# Conditional Latent Diffusion Model

![Diffusion model](imgs/conditional_diffusion_model.svg)