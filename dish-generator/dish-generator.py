import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#from huggingface_hub import login
#login(token='your-token-here')

#Careful: Downloads the specified model into .cache if used locally.
class DishGenerator:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def generate_dish_description(self, ingredients, origin, max_length=200):
        ingredients_string = ", ".join(ingredients) 
        #prompt = "Tell me an existing dish and only the dish that includes these ingredients: " + ingredients_string
        
        prompt = "When having these ingredients: "+ ingredients_string +". A possible existing dish that we can make from "+ origin +" could be:"
        
        # Tokenize the input prompt and specify padding options
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Set the pad_token_id to eos_token_id for handling padding
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Pass the attention mask here
            max_length=max_length, 
            pad_token_id=self.tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#Testing
#On Windows the model is safed in C:\Users\<YourUsername>\.cache\huggingface\transformers\

#dish_gen = DishGenerator()
#ingredients = ["Tomato Sauce","Cheese","Dough"]
#origin = "Italy"
#dish_description = dish_gen.generate_dish_description(ingredients, origin)
#print(dish_description)