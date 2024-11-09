import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

## laoding the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

## loading the json embeddings data
with open('embeddings_data.json', 'r') as file:
    courses_data = json.load(file)

## extracting the embeddings and titles
course_embeddings = [np.array(course['embedding']) for course in courses_data]
course_titles = [course['title'] for course in courses_data]

## query function
def get_query_embedding(query):
    return model.encode(query, convert_to_tensor=True)

## function to add relevance factors
def add_relevance_factors(similarities, indices):

    relevance_factor = 0.2
    ## tuple of index and score  


    enhanced_scores = []
    
    for idx in indices:
        curriculum_match = 1 if "deep learning" in course_titles[idx].lower() else 0
        enhanced_score = similarities[idx] + relevance_factor * curriculum_match
        enhanced_scores.append((idx, enhanced_score))
    
    enhanced_scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in enhanced_scores]

## user query sample
user_query = "machine learning courses with deep learning"

## converting query into embedding
query_embedding = get_query_embedding(user_query)

## calculating the similarity
cosine_similarities = cosine_similarity([np.array(query_embedding)], course_embeddings)

## getting the top 5 similar courses
top_k = 5
top_indices = cosine_similarities[0].argsort()[-top_k:][::-1]

top_indices_with_relevance = add_relevance_factors(cosine_similarities[0], top_indices)

## displayiong the results
for i in top_indices_with_relevance:
    print(f"Title: {course_titles[i]}")
    print(f"Cosine Similarity: {cosine_similarities[0][i]:.4f}")
    print("-" * 50)
