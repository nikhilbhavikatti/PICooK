import requests
import os
from PIL import Image, UnidentifiedImageError

'''
Websites used for Scraping:

https://unsplash.com/
https://www.pexels.com/
https://pixabay.com/
https://www.foodiesfeed.com/
https://www.shopify.com/stock-photos
https://commons.wikimedia.org/

'''


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

    if 'items' not in data:
        print(f"No images found for {ingredient}: Skipping...")
        return

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
