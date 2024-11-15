import json

class DishIngredientMapping():

    def __init__(self, path):
        self.path = path
        self.data = []
    
    def add(self, dish, ingredients):
        self.data.append({dish: ingredients})
    
    def get_mapping(self):
        return self.data
    
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)
    
    def load(self):
        with open(self.path, 'r') as f:
            self.data = json.load(f)
