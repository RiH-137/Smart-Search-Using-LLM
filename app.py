import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

## loading the model
model = SentenceTransformer('all-MiniLM-L6-v2')

## loading the data from the json file
with open('embeddings_data.json', 'r') as file:
    courses_data = json.load(file)

## extracting the embeddings and the course titles
course_embeddings = [np.array(course['embedding']) for course in courses_data]
course_titles = [course['title'] for course in courses_data]



## functions to get the query embedding
def get_query_embedding(query):
    return model.encode(query, convert_to_tensor=True)




## function to add relevance factors

def add_relevance_factors(similarities, indices):
    relevance_factor = 0.2  # Weight for curriculum match
    enhanced_scores = []
    
    for idx in indices:
        curriculum_match = 1 if "deep learning" in course_titles[idx].lower() else 0
        enhanced_score = similarities[idx] + relevance_factor * curriculum_match
        enhanced_scores.append((idx, enhanced_score))
    
    enhanced_scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in enhanced_scores]

## streamlit
st.title('Course Recommendation System')
st.write('Enter a search query to find relevant courses')


user_query = st.text_input('Search Query')

if user_query:
    ## conversion of the query to an embedding
    query_embedding = get_query_embedding(user_query)

    ## calculating the cosine similarity between the query and the course embeddings

    ## cosine similarity--> dot product of the query embedding and the course embedding divided by the product of the norms of the two embeddings
    cosine_similarities = cosine_similarity([np.array(query_embedding)], course_embeddings)

    ## result for the top 5 search
    top_k = 5
    top_indices = cosine_similarities[0].argsort()[-top_k:][::-1]

  
    top_indices_with_relevance = add_relevance_factors(cosine_similarities[0], top_indices)

   ## displaying the result
    st.write("### Top Course Recommendations")
    for i in top_indices_with_relevance:
        st.write(f"**Title**: {course_titles[i]}")
        st.write(f"**Cosine Similarity**: {cosine_similarities[0][i]:.4f}")
        st.write("-" * 50)
