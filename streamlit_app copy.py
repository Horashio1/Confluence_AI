import streamlit as st
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

# --- Streamlit Interface ---
def streamlit_app():
    st.title("Confluence Knowledge Base Chatbot")
    st.write("A chatbot that answers questions based on your Confluence knowledge base.")
    
    query = st.text_input("Ask your question:", placeholder="Type your query here...")
    
    if st.button("Get Answer") or query:
        if query:
            with st.spinner("Generating response..."):
                response = main(query)
                st.markdown(response)

# --- Gradio Interface ---
def create_gradio_interface():
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

if __name__ == "__main__":
    # Check if running in Streamlit environment
    if 'STREAMLIT_SCRIPT_PATH' in os.environ:
        streamlit_app()
    else:
        # Command line argument to determine which interface to use
        if len(sys.argv) > 1:
            if sys.argv[1] == "--gradio":
                # Run Gradio interface
                demo = create_gradio_interface()
                demo.launch(server_name="0.0.0.0", server_port=8080)
            elif sys.argv[1] == "--flask":
                # Run Flask webhook server only
                port = int(os.environ.get("PORT", 8080))
                print(f"Starting Flask webhook server on port {port}")
                flask_app.run(host="0.0.0.0", port=port)
            elif sys.argv[1] == "--combined":
                # Run combined Gradio + Flask using WSGI middleware
                try:
                    # First check if waitress is installed
                    import importlib.util
                    if importlib.util.find_spec("waitress") is not None:
                        from waitress import serve
                        from werkzeug.middleware.dispatcher import DispatcherMiddleware
                        
                        # Create Gradio app
                        gradio_app = create_gradio_interface()
                        
                        # Combine with Flask app
                        application = DispatcherMiddleware(gradio_app.app, {"/webhook": flask_app})
                        
                        # Serve with waitress
                        port = int(os.environ.get("PORT", 8080))
                        print(f"Starting combined app (Gradio at root, Flask at /webhook) on port {port}")
                        serve(application, host="0.0.0.0", port=port)
                    else:
                        raise ImportError("waitress module not found")
                except ImportError:
                    print("Waitress not installed. Please install with: pip install waitress")
                    print("Falling back to Gradio's built-in server...")
                    # Use Gradio's built-in server and custom route for webhook
                    demo = create_gradio_interface()
                    
                    # Mount Flask app into Gradio
                    app = demo.app
                    
                    # Add webhook route to the FastAPI app that's backing Gradio
                    @app.post("/webhook")
                    async def fastapi_webhook(request: dict):
                        if "message" in request and "text" in request["message"]:
                            incoming_text = request["message"]["text"]
                            if incoming_text.strip().startswith("Q:"):
                                query = incoming_text.strip()[2:].strip()
                                response_text = main(query)
                                return {"text": response_text}
                            else:
                                return {"text": "Please start your message with 'Q:' to ask a question."}
                        else:
                            return {"text": "Invalid message format."}, 400
                    
                    # Launch with the webhook route added
                    demo.launch(server_name="0.0.0.0", server_port=8080)
        else:
            # Default to Streamlit
            streamlit_app()