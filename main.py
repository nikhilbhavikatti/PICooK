import time
import argparse

from picook.config.ingredients_dishes import ingredients, dishes
from picook.image_retrieval.image_retrieval import get_images
from picook.zero_shot_ingredient_classifier.classifier import ImageValidator


parser = argparse.ArgumentParser(prog='PICooK',
                                 description='Helps you to find a delicious dish based on the ingredients you have at home.',
                                 epilog='Made with love in Stuttgart.')
parser.add_argument('--scrape',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Scrape images for ingredients and dishes.')
parser.add_argument('--validate',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Validate images for ingredients and dishes by using classifier.')
parser.add_argument('--move_wrong_images',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Whether wrong imges should be moved to the wrong folder or not.')
parser.add_argument('--top_k',
                    type=int,
                    required=False,
                    default=3,
                    help='Controls the sensitivity of the classifier by allowing the image class to be in the top k classes.')
parser.add_argument('--dishes',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Generates dishes by randomly sampling ingredients.')
args = parser.parse_args()


if __name__ == '__main__':
    if args.scrape:
        print("Retrieving images for ingredients")
        for list_of_ingredients in ingredients.values():
            for ingredient in list_of_ingredients:
                print(f"Retrieving images for {ingredient}...")
                get_images(ingredient)
                time.sleep(1)
        
        print("Retrieving images for dishes")
        for list_of_dishes in dishes.values():
            for item in list_of_dishes:
                print(f"Retrieving images for {item}...")
                get_images(item)
                time.sleep(1)
    
    if args.validate:
        print(f"Validating images for ingredients with top_k={args.top_k}")
        validator = ImageValidator(top_k=args.top_k)
        validator.validate_images("data/ingredients", "data/ingredients/wrong_images", move_wrong_images=args.move_wrong_images)

    if args.dishes:
        pass
