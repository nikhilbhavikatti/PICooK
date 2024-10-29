import requests
import os
from PIL import Image

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
            img = Image.open(r.raw)
            img.save(f"../data/{ingredient}_{i}.jpg")


get_images('tomatoe')