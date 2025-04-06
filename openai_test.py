from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from your .env file
env_path = find_dotenv()
load_dotenv(env_path)

# Retrieve your API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize the OpenAI client using the new interface
openai_client = OpenAI(api_key=OPENAI_API_KEY)

try:
    response = openai_client.chat.completions.create(
        model="gpt-4o", 
        # model="gpt-3.5-turbo",

        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you confirm the API is working?"}
        ],
        temperature=0.5
    )

    print("✅ API call successful!")
    print("Response:\n", response.choices[0].message.content)

except Exception as e:
    print("❌ API call failed:")
    print(e)
