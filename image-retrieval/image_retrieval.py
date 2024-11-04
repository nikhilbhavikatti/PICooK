import requests
import os
from PIL import Image, UnidentifiedImageError
import time


'''
Websites used for Scraping:

https://unsplash.com/
https://www.pexels.com/
https://pixabay.com/
https://www.foodiesfeed.com/
https://www.shopify.com/stock-photos
https://commons.wikimedia.org/

'''

# Vegetables
vegetables = [
    "artichoke", "asparagus", "aubergine", "bamboo shoots", "beetroot", "bell pepper", "bok choy",
    "broccoli", "brussels sprouts", "cabbage", "carrot", "cauliflower", "celery", "chard", "chili pepper",
    "chinese cabbage", "collard greens", "coriander leaves", "cucumber", "daikon", "eggplant", "endive",
    "fennel", "garlic", "ginger", "green beans", "green peas", "horseradish", "jicama", "kale",
    "kohlrabi", "leek", "lettuce", "mushroom", "okra", "onion", "parsley", "parsnip", "potato", "pumpkin",
    "radish", "rutabaga", "scallion", "shallot", "spinach", "sweet potato", "tomato", "turnip", "watercress",
    "yam", "zucchini"
]

# Fruits
fruits = [
    "apple", "apricot", "avocado", "banana", "blackberry", "blackcurrant", "blueberry", "boysenberry",
    "cantaloupe", "cherry", "clementine", "coconut", "cranberry", "currant", "date", "dragon fruit",
    "durian", "elderberry", "fig", "goji berry", "gooseberry", "grape", "grapefruit", "guava",
    "honeydew melon", "jackfruit", "jambul", "kiwi", "kumquat", "lemon", "lime", "lychee", "mango",
    "mulberry", "nectarine", "orange", "papaya", "passion fruit", "peach", "pear", "persimmon",
    "pineapple", "plum", "pomegranate", "quince", "raspberry", "redcurrant", "rhubarb",
    "starfruit", "strawberry", "tangerine", "watermelon"
]
# Grains and Cereals
grains_and_cereals = [
    "rice", "wheat", "oats", "corn", "quinoa", "barley", "millet", "rye", "bulgur",
    "farro", "sorghum", "couscous", "buckwheat", "spelt", "teff", "amaranth"
]

# Legumes and Pulses
legumes_and_pulses = [
    "lentils", "chickpeas", "black beans", "kidney beans", "pinto beans", "peas",
    "edamame", "split peas", "soybeans", "mung beans", "navy beans", "fava beans",
    "great northern beans", "lentil sprouts"
]

# Nuts and Seeds
nuts_and_seeds = [
    "almond", "walnut", "cashew", "hazelnut", "pistachio", "macadamia",
    "sunflower seed", "chia seed", "flaxseed", "pumpkin seed", "sesame seed",
    "pine nut", "Brazil nut", "pecan", "hemp seed", "coconut"
]

# Herbs and Spices
herbs_and_spices = [
    "basil", "cilantro", "parsley", "oregano", "thyme", "rosemary", "mint", "sage",
    "dill", "cinnamon", "cumin", "turmeric", "paprika", "black pepper", "ginger",
    "clove", "nutmeg", "bay leaf", "fennel seed", "cardamom", "allspice"
]

# Dairy Products
dairy_products = [
    "milk", "cheese", "yogurt", "butter", "cream", "sour cream", "cream cheese",
    "cottage cheese", "evaporated milk", "condensed milk", "kefir", "buttermilk"
]

# Meat and Poultry
meat_and_poultry = [
    "beef", "chicken", "pork", "lamb", "turkey", "duck", "goose", "venison",
    "bacon", "sausage", "rabbit", "quail", "pheasant"
]

# Fish and Seafood
fish_and_seafood = [
    "salmon", "tuna", "shrimp", "crab", "squid", "mackerel", "herring", "trout",
    "octopus", "tilapia", "cod", "halibut", "catfish", "scallops", "clams"
]

# Oils and Fats
oils_and_fats = [
    "olive oil", "vegetable oil", "canola oil", "butter", "ghee", "coconut oil",
    "sesame oil", "peanut oil", "avocado oil", "flaxseed oil", "grapeseed oil"
]

# Condiments and Sauces
condiments_and_sauces = [
    "ketchup", "mustard", "mayonnaise", "soy sauce", "vinegar", "hot sauce",
    "bbq sauce", "salsa", "teriyaki sauce", "pesto", "tahini", "relish"
]

# Sweeteners
sweeteners = [
    "sugar", "brown sugar", "honey", "maple syrup", "agave nectar", "corn syrup",
    "molasses", "stevia", "confectioner's sugar"
]

# Beverages
beverages = [
    "coffee", "tea", "fruit juice", "soda", "alcohol", "smoothies", "milkshake",
    "water", "coconut water", "kombucha", "herbal tea", "hot chocolate"
]

# Baked Goods and Pastry
baked_goods_and_pastry = [
    "bread", "cake", "cookie", "pastry", "muffin", "croissant", "bagel",
    "pizza", "tart", "brownie", "pancake", "waffle"
]

# Processed and Convenience Foods
processed_and_convenience_foods = [
    "frozen meals", "snacks", "canned goods", "instant noodles", "chips",
    "ready-to-eat meals", "microwave dinners", "deli meats"
]

# Fermented Foods
fermented_foods = [
    "sauerkraut", "kimchi", "yogurt", "kombucha", "miso", "tempeh",
    "pickles", "fermented soy sauce", "fermented hot sauce"
]

# Soups and Broths
soups_and_broths = [
    "chicken broth", "vegetable soup", "miso soup", "beef broth",
    "clam chowder", "tomato soup", "pumpkin soup", "lentil soup", "corn chowder"
]

def get_images(ingredient):
    api_key = os.environ['GOOGLE_API_KEY']
    search_engine_id = os.environ['GOOGLE_SEARCH_ENGINE_ID']
    r = requests.get('https://www.googleapis.com/customsearch/v1', params={'key': api_key,
                                                                           'cx': search_engine_id,
                                                                           'searchType': 'image',
                                                                           'start': 0,
                                                                           'q': ingredient})
    if r.status_code != 200:
        raise ValueError('Failed to get images')

    data = r.json()

    for i, item in enumerate(data["items"]):
        img_url = item["link"]
        print(img_url)
        r = requests.get(img_url, stream=True)

        if r.status_code == 200:
            try:
                img = Image.open(r.raw)
                save_path = os.path.join("..", "images", f"{ingredient}_{i}.jpg")
                img.save(save_path)
                print(f"Image saved to {save_path}")
            except UnidentifiedImageError:
                print(f"Skipping {img_url}: Unidentified image format")
            except Exception as e:
                print(f"Error saving {img_url}: {e}")

for item in soups_and_broths:
    print(f"Retrieving images for {item}...")
    get_images(item)

    time.sleep(1)