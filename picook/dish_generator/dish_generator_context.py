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

    def __init__(self, ingredients, model_id="meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16):
        self.model_id = model_id

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map="auto")
        
        # to match ingredients with available ingredients
        self.ingredients = ingredients
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.feature_pipeline = transformers.pipeline(
            "feature-extraction",
            model=model_id,
            torch_dtype=torch_dtype,
            device_map="auto")
        self.ingredients_embeddings = self.__get_mean_embeddings(ingredients)


    def __get_mean_embeddings(self, ingredients):
        embeddings = self.feature_pipeline([ingredient.lower() for ingredient in ingredients],
                                           tokenizer=self.tokenizer,
                                           eos_token_id=self.tokenizer.eos_token_id,
                                           return_tensors="pt")
        return torch.vstack([torch.mean(emb, axis=1) for emb in embeddings])
    

    def match_ingredients(self, ingredients, threshold=0.75):
        output = []
        embeddings = self.__get_mean_embeddings(ingredients)
        cosine_similarities = torch.nn.functional.cosine_similarity(embeddings[:, None, :], self.ingredients_embeddings[None, :, :], dim=-1)
        value, idx = torch.max(cosine_similarities, axis=1)
        print(value, idx)
        value = list(value.cpu().float().numpy())
        idx = list(idx.int().cpu().numpy())

        for idx, value in zip(idx, value):
            if value >= threshold:
                output.append(self.ingredients[idx])
        
        return output


    def generate_ingredients(self, dish, origin="", max_length=256):

        prompt = f"I would like to cook {dish} at home. Which ingredients do I have to buy?"

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

        llm_ingredients = out.split(", ")
        ingredients = self.match_ingredients(llm_ingredients)

        return ingredients, llm_ingredients
