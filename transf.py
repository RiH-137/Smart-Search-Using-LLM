import pandas as pd

# Load the CSV file
df = pd.read_csv('course_titles.csv')  
sentences = df['title'].tolist()  # Replace with your sentence column name


from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different pre-trained model if needed

# Convert sentences to embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)



import json
import numpy as np

# Create a list of dictionaries containing metadata and embeddings
data = []
for i, sentence in enumerate(sentences):
    # Example of adding course metadata along with embeddings
    item = {
        'title': df['title'][i],  # Replace with your course title column
        'embedding': embeddings[i].tolist()  # Convert the tensor to a list
    }
    data.append(item)

# Save data to JSON file
with open('embeddings_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
