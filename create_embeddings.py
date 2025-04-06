#!/usr/bin/env python3
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# ============================
# Global Configuration
# ============================
CSV_FILE = "./conf_data_BigData.csv"         # Define your CSV file path here
INDEX_NAME = "big-data"                      # Define your Pinecone index name here
OVERWRITE = True                             # Set True to overwrite an existing index, False to add/upsert
LIMIT = 1900                                    # Limit the number of records to process for testing (set None for no limit)
SPACE_KEY = "BDDS"                           # (Optional) Space key, if needed in your data prep

# ============================
# Environment Setup
# ============================
env_path = find_dotenv()
print(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Retrieve required environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "https://pickme.atlassian.net/wiki")

# ============================
# Import Utility Functions
# ============================
from utils.pinecone_logic import delete_pinecone_index, get_pinecone_index, upsert_data
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df

# ============================
# Main Function
# ============================
def main():
    # Connect to (or create) the Pinecone index.
    try:
        index, index_created = get_pinecone_index(INDEX_NAME)
        if OVERWRITE:
            print(f"Overwriting existing index: {INDEX_NAME}")
            delete_pinecone_index(INDEX_NAME)
            index, index_created = get_pinecone_index(INDEX_NAME)
        else:
            stats = index.describe_index_stats()
            print(f"Index {INDEX_NAME} exists with {stats.total_vector_count} vectors. New embeddings will be added/upserted.")
    except Exception as e:
        print(f"Error accessing Pinecone index: {e}")
        return

    # Load CSV file into a DataFrame using the helper utility.
    try:
        # Note: The working code creates an initial empty DataFrame with expected columns.
        df = pd.DataFrame(columns=['id', 'tiny_link', 'content'])
        df = import_csv(df, CSV_FILE, max_rows=2000)
    except Exception as e:
        print(f"Error importing CSV: {e}")
        return

    # Optionally limit the number of records for testing.
    if LIMIT is not None:
        df = df.head(LIMIT)
        print(f"Processing first {LIMIT} records for testing purposes.")

    # Clean the DataFrame so it conforms to the Pinecone schema.
    try:
        df = clean_data_pinecone_schema(df)
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return

    # Generate embeddings and add them to the DataFrame.
    try:
        df = generate_embeddings_and_add_to_df(df, OPENAI_EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # Upsert the processed data to the Pinecone index.
    print("Upserting embeddings to Pinecone...")
    try:
        upsert_data(index, df)
    except Exception as e:
        print(f"Error upserting data to Pinecone: {e}")
        return

    stats = index.describe_index_stats()
    print(f"Successfully uploaded {stats.total_vector_count} vectors to index '{INDEX_NAME}'.")

if __name__ == "__main__":
    main()
