import json

class DishIngredientMapping():

    def __init__(self, path):
        self.path = path
        self.data = []
        self.metadata = {}
    
    def add(self, dish, ingredients, origin="", ingredients_llm=""):
        self.data.append({"dish_name": dish, "ingredients": ingredients, "origin:": origin, "ingredients_llm": ingredients_llm})
    
    def get_mapping(self):
        return self.data
    
    def set_metadata(self, metadata):
        self.metadata = metadata
    
    def get_metadata(self):
        return self.metadata
    
    def save(self):
        with open(self.path, 'w', encoding='utf8') as f:
            json.dump({"metadata": self.metadata, "data": self.data}, f, indent=4, ensure_ascii=False)
    
    def load(self):
        with open(self.path, 'r', encoding='utf8') as f:
            json_file = json.load(f)
            self.data = json_file["data"]
            self.metadata = json_file["metadata"]
