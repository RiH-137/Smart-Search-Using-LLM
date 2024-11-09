import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the embeddings data from the JSON file
with open('embeddings_data.json', 'r') as file:
    courses_data = json.load(file)

# Extract the embeddings and titles
course_embeddings = [np.array(course['embedding']) for course in courses_data]
course_titles = [course['title'] for course in courses_data]

# Function to get query embedding
def get_query_embedding(query):
    return model.encode(query, convert_to_tensor=True)

# Function to add relevance factors
def add_relevance_factors(similarities, indices):
    relevance_factor = 0.2  # Weight for curriculum match
    enhanced_scores = []
    
    for idx in indices:
        curriculum_match = 1 if "deep learning" in course_titles[idx].lower() else 0
        enhanced_score = similarities[idx] + relevance_factor * curriculum_match
        enhanced_scores.append((idx, enhanced_score))
    
    enhanced_scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in enhanced_scores]

# Example user query
user_query = "machine learning courses with deep learning"

# Convert the user query to embedding
query_embedding = get_query_embedding(user_query)

# Calculate cosine similarities
cosine_similarities = cosine_similarity([np.array(query_embedding)], course_embeddings)

# Get top 5 most similar courses
top_k = 5
top_indices = cosine_similarities[0].argsort()[-top_k:][::-1]

# Apply relevance factors (optional)
top_indices_with_relevance = add_relevance_factors(cosine_similarities[0], top_indices)

# Display the top results
for i in top_indices_with_relevance:
    print(f"Title: {course_titles[i]}")
    print(f"Cosine Similarity: {cosine_similarities[0][i]:.4f}")
    print("-" * 50)
