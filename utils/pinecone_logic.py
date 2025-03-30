from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import ast
import os

#Global variables
pinecone = None

def initialize_pinecone_client():
    """Initialize the Pinecone client with API key."""
    global pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    pinecone = Pinecone(api_key=api_key)
    return pinecone

# delete index
def delete_pinecone_index(index_name):
    if pinecone is None:
        initialize_pinecone_client()
    print(f"Deleting index '{index_name}' if it exists.")
    try:
        pinecone.delete_index(index_name)
        print(f"Index '{index_name}' successfully deleted.")
    except Exception as e:
        print(f"index '{index_name}' not found no action taken.")


# create index if needed
def get_pinecone_index(index_name):
    print(f"Checking if index {index_name} exists.")
    index_created = False
    
    # Ensure Pinecone client is initialized
    if pinecone is None:
        initialize_pinecone_client()
        
    if index_name in [index.name for index in pinecone.list_indexes()]:
        print(f"Index {index_name} already exists, good to go.")
        index = pinecone.Index(index_name)
    else:
        print(f"Index {index_name} does not exist, need to create it.")
        index_created = True
        pinecone.create_index(
            name=index_name, 
            dimension=1536, 
            metric='cosine', 
            spec=ServerlessSpec(cloud='aws', region='us-east-1'))
            
        print(f"Index {index_name} created.")

        index = pinecone.Index(index_name)
    return index, index_created


# Function to upsert data
def upsert_data(index, df):
    print("Start: Upserting data to Pinecone index")
    prepped = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        meta = ast.literal_eval(row['metadata'])
        prepped.append({'id': row['id'], 
                        'values': row['values'],
                        'metadata': meta})
        if len(prepped) >= 200: # batching upserts
            index.upsert(prepped)
            prepped = []

    # Upsert any remaining entries after the loop
    if len(prepped) > 0:
        index.upsert(prepped)
    
    print("Done: Data upserted to Pinecone index")
    return index

