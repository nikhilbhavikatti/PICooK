import json

class DishIngredientMapping():

    def __init__(self, path):
        self.path = path
        self.data = []
        self.metadata = {}
    
    def add(self, dish, ingredients, origin):
        self.data.append({"dish_name": dish, "ingredients": ingredients, "origin:": origin})
    
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
            json = json.load(f)
            self.data = json["data"]
            self.metadata = json["metadata"]
