import torch
import transformers

from huggingface_hub import login

from ingredients_dishes import ingredients
import json
from dish_generator_context import InverseDishGenerator
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu,SmoothingFunction
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

login(token='Token')

possible_ingredients = [item for sublist in ingredients.values() for item in sublist]
print(possible_ingredients)

generator = InverseDishGenerator(possible_ingredients)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Tokenization function
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def BLEU_score(reference, candidate):

    # Preprocess
    reference_tokenized = [tokenize(reference)]
    candidate_tokenized = tokenize(candidate)

    #print(reference)
    #print(candidate)

    # BLEU score computation
    score = sentence_bleu(reference_tokenized, candidate_tokenized, smoothing_function=SmoothingFunction().method3)
    #print(f"BLEU score: {score}")
    return score

    #####################


def cosine_sim(reference, candidate):
    # Generate embeddings
    ref_embedding = model.encode(reference, convert_to_tensor=True)
    gen_embedding = model.encode(candidate, convert_to_tensor=True)

    # Move embeddings to CPU and convert to NumPy
    ref_embedding_np = ref_embedding.cpu().numpy()
    gen_embedding_np = gen_embedding.cpu().numpy()

    # Compute cosine similarity
    similarity = cosine_similarity(ref_embedding_np.reshape(1, -1), gen_embedding_np.reshape(1, -1))[0][0]

    #print(f"Cosine Similarity: {similarity}")
    return float(similarity)

def clean_ingredient(ingredient):
    # Remove everything inside parentheses, including the parentheses
    no_parentheses = re.sub(r"\(.*?\)", "", ingredient)
    # Remove everything after the first comma
    no_commas = no_parentheses.split(",")[0].strip()
    # Remove numbers of any format (e.g., 1, 0.5, ½)
    no_numbers = re.sub(r"\b\d+(\.\d+)?|\b\d+/\d+\b", "", no_commas).strip()
    # Remove specific strings like 'tbsp' and 'tsp'
    cleaned = re.sub(r"\b(tbsp|tsp)\b", "", no_numbers, flags=re.IGNORECASE).strip()
    return cleaned
    
    
sum_cosine = 0;
sum_BLEU = 0;

BLEU_array = []
cosine_array = []
ingredients_array = []
ingredients_output_array = []

for i in range(len(data)):
    print(i)

    name = data[i]['name']
    print(name)
    ingredients = data[i]['ingredients']

    cleaned_ingredients = [clean_ingredient(item) for item in ingredients]

    ingredients_generated = generator.generate_ingredients(data[i]['name'])

    ingredients_array.append(cleaned_ingredients)
    ingredients_output_array.append(ingredients_generated[1])

    ingredients_generated_string = ",".join(ingredients_generated[1])
    ingredients_reference_string = ",".join(cleaned_ingredients)

    #print(ingredients_generated_string)
    #print(ingredients_reference_string)

    BLEU_value = BLEU_score(ingredients_reference_string, ingredients_generated_string)
    cosine_value = cosine_sim(ingredients_reference_string, ingredients_generated_string)

    #print(cosine_value)

    BLEU_array.append(BLEU_value)
    cosine_array.append(cosine_value)

    sum_BLEU += BLEU_value
    sum_cosine += cosine_value


print(sum_cosine/len(data))
print(sum_BLEU/len(data))

# Create a list of dictionaries to store the data
json_data = []
for i in range(len(data)):
    json_data.append({
        'name': data[i]['name'],
        'true_ingredients': ingredients_array[i],
        'generated_ingredients': ingredients_output_array[i],
        'BLEU_score': BLEU_array[i],
        'cosine_similarity': cosine_array[i]
    })

# Write to a JSON file
with open('generated_dishes.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print("JSON file created successfully!")

#Average cosine sim: 0.7833148607885767
#Average BLEU score: 0.02803329007560598