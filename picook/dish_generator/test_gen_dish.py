from huggingface_hub import login

import json
from dish_generator_context import DishGenerator
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu,SmoothingFunction
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

login(token='token')

generator = DishGenerator()
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
    return similarity
    
# Used Recipes3k
with open('recipes.json', 'r') as file:
    data = json.load(file)

# 'data' is now a Python dictionary
print(data[1]['ingredients'])
print(data[1]['name'])
print(len(data))

sum_cosine = 0;
sum_BLEU = 0;

BLEU_array = []
cosine_array = []
name_output_array = []

for i in range(len(data)):
    print(i)
    name_output = generator.generate_dish(data[i]['ingredients'])
    name_output_array.append(name_output)

    #print(data[i]['name'])
    #print(name_output)

    BLEU_value = BLEU_score(data[i]['name'], name_output)
    cosine_value = cosine_sim(data[i]['name'], name_output)
    
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
        'generated': name_output_array[i],
        'BLEU_score': BLEU_array[i],
        'cosine_similarity': cosine_array[i]
    })

# Write to a JSON file
with open('generated_dishes.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print("JSON file created successfully!")

# Average cosine_similarity: 0.5710595061167812
# Average BLEU: 0.12136529939034955