import pandas as pd
import json
import numpy as np

from sentence_transformers import SentenceTransformer

## loading the data from the csv file
df = pd.read_csv('course_titles.csv')  
sentences = df['title'].tolist()  # Replace with your sentence column name



## laoding the model    
model = SentenceTransformer('all-MiniLM-L6-v2') 

## converting the sentences to embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)





##creating a dictionary to store the embeddings and the course title
data = []
for i, sentence in enumerate(sentences):
    
    item = {
        'title': df['title'][i],  
        'embedding': embeddings[i].tolist()  
    }
    data.append(item)

## saving data into a json file
with open('embeddings_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
