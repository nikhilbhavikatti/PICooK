# Foundation Modles Project PICooK


# Image Retrieval
```bash
export GOOGLE_API_KEY="<KEY>"
export GOOGLE_SEARCH_ENGINE_ID="<ID>"
```

# DISH GENERATION
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