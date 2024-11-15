import torch
import transformers


class DishGenerator:
    """
    Dish generator that provides system context to the LLM.
    """

    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16):
        self.model_id = model_id

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map="auto")


    def generate_dish(self, ingredients, origin="", max_length=256):
        ingredients_string = ", ".join(ingredients)
        if (origin == ""):
            prompt = f"When having these ingredients: {ingredients_string}. One possible existing dish that we can make could be?"
        else:
            prompt = f"When having these ingredients: {ingredients_string}. One possible existing dish that we can make from {origin} could be?"

        # Set context for LLM
        messages = [{"role": "system", "content": "You are a chef who outputs a single dish given a list of ingredients if asked. You are very creative and allowed to use other ingredients. You only respond with the name of the dish and not with any additional text."},
                    {"role": "user", "content": prompt}]
        
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        out = self.pipeline(
            messages,
            max_new_tokens=max_length,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )[0]["generated_text"][-1]["content"]

        return out


class InverseDishGenerator:
    """
    Inverse dish generator that provides system context to the LLM.
    """

    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16):
        self.model_id = model_id

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map="auto")


    def generate_ingredients(self, dish, origin="", max_length=256):

        prompt = f"I would like to cook {dish} at home. Which ingredients do I need to buy?"

        # Set context for LLM
        messages = [{"role": "system", "content": "You are a chef who outputs a list of ingredients to cook a dish if asked. You only reply with the most important ingredients and without any additional information to the ingredients. You only respond with a list where each ingredient is separated by a comma. You do not respond with any additional text."},
                    {"role": "user", "content": prompt}]
        
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        out = self.pipeline(
            messages,
            max_new_tokens=max_length,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )[0]["generated_text"][-1]["content"]

        return out