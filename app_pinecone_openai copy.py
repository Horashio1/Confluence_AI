import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import gradio as gr
from utils.openai_logic import get_embeddings, create_prompt, add_prompt_messages, get_chat_completion_messages, create_system_prompt
import sys
from typing import Optional

# load environment variables
env_path = find_dotenv()
print(f"Loading .env file from: {env_path}")
load_dotenv(env_path)
print(f"PINECONE_API_KEY loaded: {'PINECONE_API_KEY' in os.environ}")
print(f"PINECONE_API_KEY value: {os.getenv('PINECONE_API_KEY')[:10]}...")  # Only print first 10 chars for security

# Import Pinecone utilities after environment variables are loaded
from utils.pinecone_logic import delete_pinecone_index, get_pinecone_index, upsert_data
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df

# Configuration
MODEL_FOR_OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MODEL_FOR_OPENAI_CHAT = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo-0125")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "test1")
CSV_FILE = os.getenv("CSV_FILE_PATH", "./conf_data.csv")

# Global variables
index = None
df = None

def initialize_pinecone():
    """Initialize Pinecone index and load data if needed."""
    global index, df
    
    try:
        index, index_created = get_pinecone_index(INDEX_NAME)
        
        # Add verification of index stats
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        # If no vectors or new index created, load data
        if total_vectors == 0 or index_created:
            print(f"No vectors found in index or new index created. Loading data from {CSV_FILE}")
            try:
                df = pd.DataFrame(columns=['id', 'tiny_link', 'content'])
                df = import_csv(df, CSV_FILE, max_rows=2000)
                df = clean_data_pinecone_schema(df)
                df = generate_embeddings_and_add_to_df(df, MODEL_FOR_OPENAI_EMBEDDING)
                upsert_data(index, df)
                
                # Verify data was uploaded
                stats = index.describe_index_stats()
                if stats.total_vector_count == 0:
                    raise Exception("Failed to upload vectors to Pinecone index")
                print(f"Successfully uploaded {stats.total_vector_count} vectors to index")
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                return False
            
        return True
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return False

def extract_info(data):
    """Extract source, title, and score information from matches."""
    try:
        extracted_info = []
        confluence_base_url = os.getenv('CONFLUENCE_BASE_URL', 'https://pickme.atlassian.net/wiki')
        for match in data['matches']:
            page_id = match['metadata'].get('page_id')
            title = match['metadata'].get('title', 'No title')  # Get title from metadata
            score = match['score']
            
            if page_id:
                source = f"{confluence_base_url}/spaces/BDDS/pages/{page_id}"
            else:
                source = match['metadata']['source']
                
            extracted_info.append((source, title, score))
        return extracted_info
    except Exception as e:
        print(f"Error extracting info: {str(e)}")
        return []

def main(query: str) -> Optional[str]:
    """Main function to process queries and return responses."""
    try:
        print("Start: Main function")
        
        if not initialize_pinecone():
            return "Error: Failed to initialize Pinecone. Please check your configuration."

        embed = get_embeddings(query, MODEL_FOR_OPENAI_EMBEDDING)
        res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
        
        messages = []
        system_prompt = create_system_prompt()
        prompt = create_prompt(query, res)
        messages = add_prompt_messages("system", system_prompt, messages)
        messages = add_prompt_messages("user", prompt, messages)
        response = get_chat_completion_messages(messages, MODEL_FOR_OPENAI_CHAT)
        
        print('-' * 80)
        extracted_info = extract_info(res)
        validated_info = []
        for info in extracted_info:
            source, title, score = info
            validated_info.append(f"[{title}]({source})    Score: {score}")

        validated_info_str = "\n\n".join(validated_info)
        final_output = f"{response}\n\n{validated_info_str}"
        print(final_output)
        print('-' * 80)
        return final_output
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    main("What is your contact information?")

# Gradio interface
def create_gradio_interface():
    gr.close_all()
    demo = gr.Interface(
        fn=main,
        inputs=[gr.Textbox(label="Hello, my name is Aiden, your customer service assistant, how can i help?", lines=1, placeholder="")],
        outputs=[gr.Markdown(label="response")],
        title="Confluence Knowledge Base Chatbot",
        description="A question and answering chatbot that answers questions based on your confluence knowledge base. Note: anything that was tagged internal_only has been removed",
        allow_flagging="never"
    )
    return demo

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--gradio":
    demo = create_gradio_interface()
    demo.launch(server_name="localhost", server_port=8888)    

#create Gradio interface for the chatbot
gr.close_all()
demo = gr.Interface(
    fn=main,
    inputs=[gr.Textbox(label="Hello, my name is Aiden, your customer service assistant, how can i help?", lines=1, placeholder="")],
    outputs=[gr.Markdown(label="response")],
    title="Customer Service Assistant",
    description="A question and answering chatbot that answers questions based on your confluence knowledge base. Note: anything that was tagged internal_only has been removed",
    allow_flagging="never"
)
demo.launch(server_name="localhost", server_port=8888)  