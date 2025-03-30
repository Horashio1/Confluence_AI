import pandas as pd
import json
from tqdm.auto import tqdm
from utils.openai_logic import create_embeddings
import os, sys
import numpy as np

# Function to get dataset
def import_csv(df, csv_file, max_rows):
    print("Start: Getting dataset")

    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at: {csv_file}")
    
    try:
        # Add 'title' to usecols
        df = pd.read_csv(csv_file, usecols=['id', 'tiny_link', 'content', 'title'], nrows=max_rows)
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("No data found in the CSV file.")
            
        # Update required columns to include title
        required_columns = {'id', 'tiny_link', 'content', 'title'}
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            raise ValueError(f"CSV file is missing required columns: {missing_columns}")
            
        return df
        
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")


def clean_data_pinecone_schema(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
        
    # Filter out rows where 'content' is empty or NaN
    df = df[df['content'].notna() & (df['content'] != '')]
    
    if df.empty:
        raise ValueError("No valid data found in the CSV file after filtering empty content.")
    
    # Proceed with the function's main logic
    df['id'] = df['id'].astype(str)
    df.rename(columns={'tiny_link': 'source'}, inplace=True)
    
    # Get Confluence base URL from environment variable
    confluence_base_url = os.getenv('CONFLUENCE_BASE_URL', 'https://pickme.atlassian.net/wiki')
    
    # Modify the metadata creation to include the title
    df['metadata'] = df.apply(
        lambda row: json.dumps({
            'source': f"{confluence_base_url}/spaces/BDDS/pages/{row['id']}",
            'text': row['content'],
            'page_id': row['id'],  # Store the page ID separately for future reference
            'title': row['title']  # Add title to metadata
        }), 
        axis=1
    )
    df = df[['id', 'metadata']]
    print("Done: Dataset retrieved")
    return df


def chunk_text(text, max_chunk_size=8000):
    """Split text into chunks while preserving sentence boundaries."""
    # Split into sentences (rough approximation)
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # Rough estimation of tokens (1 token ≈ 4 chars)
        sentence_size = len(sentence) // 4
        
        if current_size + sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks


# Function to generate embeddings and add to DataFrame
def generate_embeddings_and_add_to_df(df, model_emb):
    print("Start: Generating embeddings and adding to DataFrame")
    if df is None or 'metadata' not in df.columns:
        print("Error: DataFrame is None or missing 'metadata' column.")
        return None
    
    df['values'] = None

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            content = row['metadata']
            meta = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for row {index}: {e}")
            continue

        text = meta.get('text', '')
        if not text:
            print(f"Warning: Missing 'text' in metadata for row {index}. Skipping.")
            continue

        try:
            # Split text into chunks if it's too large
            chunks = chunk_text(text)
            chunk_embeddings = []
            
            # Generate embedding for each chunk
            for chunk in chunks:
                response = create_embeddings(chunk, model_emb)
                if response:
                    chunk_embeddings.append(response)
            
            if chunk_embeddings:
                # Average the embeddings from all chunks
                final_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                df.at[index, 'values'] = final_embedding
            else:
                print(f"Warning: No embeddings generated for row {index}")
                
        except Exception as e:
            print(f"Error generating embedding for row {index}: {e}")

    print("Done: Generating embeddings and adding to DataFrame")
    return df.dropna(subset=['values'])  # Remove rows where embedding failed

