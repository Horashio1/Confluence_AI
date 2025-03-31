import os
import sys
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
import gradio as gr
from typing import Optional

# Import your custom functions from your utils
from utils.openai_logic import (
    get_embeddings, create_prompt, add_prompt_messages,
    get_chat_completion_messages, create_system_prompt
)
from utils.pinecone_logic import delete_pinecone_index, get_pinecone_index, upsert_data
from utils.data_prep import import_csv, clean_data_pinecone_schema, generate_embeddings_and_add_to_df

# Load environment variables
env_path = find_dotenv()
print(f"Loading .env file from: {env_path}")
load_dotenv(env_path)
print(f"PINECONE_API_KEY loaded: {'PINECONE_API_KEY' in os.environ}")
print(f"PINECONE_API_KEY value: {os.getenv('PINECONE_API_KEY')[:10]}...")  # For security

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
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        if total_vectors == 0 or index_created:
            print(f"No vectors found in index or new index created. Loading data from {CSV_FILE}")
            try:
                df = pd.DataFrame(columns=['id', 'tiny_link', 'content'])
                df = import_csv(df, CSV_FILE, max_rows=2000)
                df = clean_data_pinecone_schema(df)
                df = generate_embeddings_and_add_to_df(df, MODEL_FOR_OPENAI_EMBEDDING)
                upsert_data(index, df)
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
            title = match['metadata'].get('title', 'No title')
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

# --- Flask Webhook Integration ---
flask_app = Flask(__name__)

@flask_app.route("/webhook", methods=["POST"])
def webhook():
    """
    Google Chat sends events to this endpoint.
    If the message text starts with "Q:", process it as a query.
    """
    event = request.get_json()
    print("Received event:", event)
    if "message" in event and "text" in event["message"]:
        incoming_text = event["message"]["text"]
        if incoming_text.strip().startswith("Q:"):
            query = incoming_text.strip()[2:].strip()  # Remove the "Q:" prefix
            response_text = main(query)
            return jsonify({"text": response_text})
        else:
            return jsonify({"text": "Please start your message with 'Q:' to ask a question."})
    else:
        return jsonify({"text": "Invalid message format."}), 400

# --- Gradio Interface ---
def create_gradio_interface():
    gr.close_all()
    demo = gr.Interface(
        fn=main,
        inputs=[gr.Textbox(label="Ask your question:", lines=1, placeholder="Type your query here...")],
        outputs=[gr.Markdown(label="Response")],
        title="Confluence Knowledge Base Chatbot",
        description=("A chatbot that answers questions based on your Confluence knowledge base. "
                     "For Google Chat, prefix your query with 'Q:'"),
        allow_flagging="never"
    )
    return demo

# --- Mounting Gradio as the Default App with Flask Webhook on /webhook ---
if __name__ == "__main__":
    # Create the Gradio interface WSGI app
    gradio_app = create_gradio_interface().app  # This is a WSGI app

    # Mount the Gradio interface as the default (root) app
    # Mount the Flask webhook app under /webhook so it's still accessible.
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    application = DispatcherMiddleware(gradio_app, {"/webhook": flask_app})

    # Use Waitress to serve the combined app (install via `pip install waitress`)
    try:
        from waitress import serve
        port = int(os.environ.get("PORT", 8080))
        print(f"Starting combined app (Gradio at root, Flask at /webhook) on port {port}")
        serve(application, host="0.0.0.0", port=port)
    except ImportError:
        # Fallback to Flask's built-in server for development purposes only
        port = int(os.environ.get("PORT", 8080))
        print(f"Starting Flask app on port {port}")
        gradio_app.run(host="0.0.0.0", port=port)
