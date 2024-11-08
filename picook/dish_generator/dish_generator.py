import torch
import gc
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
#from huggingface_hub import login
#login(token='your_token_here')

#Careful: Downloads the specified model into .cache if used locally.
class DishGenerator:
  def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16):
    self.model_id = model_id
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set pad token if not already set
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # Load model and move to GPU if available
    self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    if torch.cuda.is_available():
      self.model = self.model.to("cuda")

  def generate_dish_description(self, ingredients, origin = "", max_length=300):
    ingredients_string = ", ".join(ingredients)
    if(origin == ""):
      prompt = f"When having these ingredients: {ingredients_string}. One possible existing dish that we can make could be:"
    else:
      prompt = f"When having these ingredients: {ingredients_string}. One possible existing dish that we can make from {origin} could be:"

    # Tokenize input prompt
    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
    # Move input tensors to GPU if available
    if torch.cuda.is_available():
      inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate the output
    outputs = self.model.generate(
      inputs['input_ids'],
      attention_mask=inputs['attention_mask'],
      max_length=max_length,
      pad_token_id=self.tokenizer.eos_token_id
    )
        
    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
      
  def get_first_sentence(self, description):
    subtext = description.split(":", 2)[-1].strip()
    period_index = subtext.find(".")
    return subtext[:period_index] if period_index != -1 else subtext

##Testing
#dish_gen = DishGenerator()
#ingredients = ["Salami","Cheese","Dough","Tomato Sauce"]
#origin = "Italy"

#dish_description = dish_gen.generate_dish_description(ingredients, origin)
#dish = dish_gen.get_first_sentence(dish_description)
#print(dish_description)
#print(dish)

##Empty GPU memory
#del dish_gen
#del dish_description
#gc.collect()
#torch.cuda.empty_cache()