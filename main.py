import time
import argparse
import random

from picook.config.ingredients_dishes import ingredients, dishes, origins
from picook.config.utils import DishIngredientMapping
from picook.image_retrieval.image_retrieval import get_images
from picook.zero_shot_ingredient_classifier.classifier import ImageValidator
from picook.dish_generator.dish_generator_context import DishGenerator, InverseDishGenerator
from picook.dataset_generator.dataset_generator import DatasetGenerator

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
parser.add_argument('--evaluate',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Evaluate the classifier using a labeled dataset.')
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
parser.add_argument('--min_ingredients',
                    type=int,
                    required=False,
                    default=4,
                    help='Minimal number of ingredients to sample.')
parser.add_argument('--max_ingredients',
                    type=int,
                    required=False,
                    default=10,
                    help='Maximal number of ingredients to sample.')
parser.add_argument('--num_dishes',
                    type=int,
                    required=False,
                    default=256,
                    help='Number of dishes to generate.')
parser.add_argument('--use_origin',
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Whether a dish from a specific origin should be generated.')
parser.add_argument('--ingredients',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Generates ingredients of the given dishes.')
parser.add_argument('--dataset',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Generates the dataset by combining the mapping with the images.')
parser.add_argument('--split',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Splits the generated dataset into train and test.')


if __name__ == '__main__':
    args = parser.parse_args()

    # Scrape images of ingredients and dishes
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
    
    # Validate scraped images using zero-shot classifier
    if args.validate:
        print(f"Validating images for ingredients with top_k={args.top_k}")
        validator = ImageValidator(top_k=args.top_k)
        validator.validate_images("data/ingredients", "data/ingredients/wrong_images", move_wrong_images=args.move_wrong_images)
        validator.validate_images("data/dishes", "data/dishes/wrong_images", move_wrong_images=args.move_wrong_images)

    # Evaluate the classifier
    if args.evaluate:
        print("Evaluating the classifier...")
        validator = ImageValidator(top_k=args.top_k)

        # Paths for evaluation
        image_directory = "data/ingredients"  # Path to your labeled image dataset
        labels_json = "data/ingredients_labels.json"  # Path to the JSON file with labels
        wrong_images_directory = "data/ingredients/wrong_images"  # Directory for wrong images

        # Call evaluate_images and print the results
        metrics = validator.evaluate_images(
            image_directory,
            labels_json,
            wrong_images_directory,
            move_wrong_images=args.move_wrong_images
        )

        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    # Generate dishes by randomly sampling ingredients
    if args.dishes:
        generator = DishGenerator()
        mapping = DishIngredientMapping("data/dish_ingredient_mapping.json")
        mapping.set_metadata({"forward_mapping": True,
                              "min_ingredients": args.min_ingredients,
                              "max_ingredients": args.max_ingredients,
                              "use_origin": args.use_origin})
        all_ingredients = [ingredient for list_of_ingredients in ingredients.values() for ingredient in list_of_ingredients]
        for i in range(args.num_dishes):
            num_ingredients = int(random.uniform(args.min_ingredients, args.max_ingredients))
            ingredients = random.sample(all_ingredients, num_ingredients)
            if args.use_origin:
                origin = random.sample(origins, 1)[0]
                dish = generator.generate_dish(ingredients, origin)
            else:
                origin = ""
                dish = generator.generate_dish(ingredients)
            print(ingredients, origin, dish)
            mapping.add(dish, ingredients, origin=origin)
        mapping.save()
    
    # Generate ingredients for the given dishes
    if args.ingredients:
        all_ingredients = [ingredient for list_of_ingredients in ingredients.values() for ingredient in list_of_ingredients]
        all_dishes = [dish for list_of_dishes in dishes.values() for dish in list_of_dishes]
        inverse_generator = InverseDishGenerator(all_ingredients)
        mapping = DishIngredientMapping("data/dish_ingredient_mapping_inverse.json")
        mapping.set_metadata({"forward_mapping": False})
        for dish in all_dishes[:args.num_dishes]:
            ingredients, ingredients_llm = inverse_generator.generate_ingredients(dish)
            if len(ingredients) > 0:
                mapping.add(dish, ingredients, ingredients_llm=ingredients_llm)
            print(dish, ingredients, ingredients_llm)
        mapping.save()

    # Make the dataset
    if args.dataset:
        mapping = DishIngredientMapping("data/dish_ingredient_mapping_inverse_all.json")
        mapping.load()
        generator = DatasetGenerator(mapping.get_mapping(), data_dir="data")
        generator.generate()

        # Also do the train and test split
        if args.split:
            generator.split()
    
    # Train and test split
    if args.split:
        mapping = DishIngredientMapping("data/dish_ingredient_mapping_inverse_all.json")
        mapping.load()
        generator = DatasetGenerator(mapping.get_mapping(), data_dir="data")
        generator.split()
