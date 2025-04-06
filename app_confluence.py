import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import pandas as pd
import sys
from bs4 import BeautifulSoup
from tqdm import tqdm  # Make sure to import tqdm at the top of your script

# Load environment variables from .env file
load_dotenv()
confluence_domain = os.getenv("confluence_domain")
api_key = os.getenv("CONF_API_KEY")

# Ensure confluence_domain has the proper scheme
if not confluence_domain.startswith(('http://', 'https://')):
    confluence_domain = f'https://{confluence_domain}'

# Set your Confluence details here
space_key = 'BC'  # Big data and data science space
# BC - Product Development space
# BDDS - Big Data and Data Science


# Function to fetch pages from Confluence
def fetch_pages(start, limit):
    # Updated expand parameter to include ancestors and history for parent page and published date info
    url = f'{confluence_domain}/wiki/rest/api/content?spaceKey={space_key}&start={start}&limit={limit}&expand=title,ancestors,history'
    json_data = api_call(url)
    if json_data is not None:
        return json_data
    else:
        print("Failed to fetch pages.")
        return None
    
# Function to make an API call
def api_call(url):
    try:
        # Use Basic Authentication with email and API token
        auth = HTTPBasicAuth(os.getenv("CONF_EMAIL"), api_key)
        headers = {
            'Accept': 'application/json'
        }
        print(f"\nMaking API call to: {url}")
        print(f"Using email: {os.getenv('CONF_EMAIL')}")
        print(f"API key present: {'Yes' if api_key else 'No'}")
        
        response = requests.get(url, auth=auth, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"Error: Page not found. URL: {url}")
            print(f"Response content: {response.text}")
        elif response.status_code == 401:
            print("Error: Authentication failed. Please check your email and API token.")
            print(f"Response content: {response.text}")
        elif response.status_code == 403:
            print("Error: Access forbidden. Please check your permissions and API token.")
            print(f"Response content: {response.text}")
        elif response.status_code == 500:
            print("Error: Internal server error.")
            print(f"Response content: {response.text}")
        else:
            print(f"Failed to get pages: HTTP status code {response.status_code}")
            print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        print(f"Request URL: {url}")
        print(f"Request headers: {headers}")

    return None


# Function to fetch labels from Confluence
def fetch_labels(page_id):
    url = f'{confluence_domain}/wiki/rest/api/content/{page_id}/label'
    json_data = api_call(url)

    if json_data:
        try:
            internal_only = False
            for item in json_data.get("results", []):
                if item.get("name") == 'internal_only':
                    internal_only = True

            return internal_only
        except KeyError:
            print("Error processing JSON data.")
            return None
    else:
        print("Failed to fetch labels.")
        return None


# Function to fetch page content from Confluence
def fetch_page_content(page_id):
    url = f'{confluence_domain}/wiki/rest/api/content/{page_id}?expand=body.storage'
    json_data = api_call(url)

    if json_data:
        try:
            return json_data['body']['storage']['value']
        except KeyError:
            print("Error: Unable to access page content in the returned JSON.")
            return None
    else:
        print("Failed to fetch page content.")
        return None
    

# Function to create an empty DataFrame    
def create_dataframe():
    try:
        # Added 'parent_page_name' and 'published_date' columns to the metadata
        columns = ['id', 'type', 'status', 'tiny_link', 'title', 'parent_page_name', 'published_date', 'content', 'is_internal']
        df = pd.DataFrame(columns=columns)
        return df
    except Exception as e:
        print(f"An error occurred while creating the DataFrame: {e}")
        return None


# Function to add all pages to the DataFrame
def add_all_pages_to_dataframe(df, all_pages):
    if not isinstance(df, pd.DataFrame):
        print("Error: The first argument must be a pandas DataFrame.")
        return None

    if not isinstance(all_pages, list):
        print("Error: The second argument must be a list.")
        return None

    for page in all_pages:
        try:
            # Extract the parent's page name from ancestors if available.
            parent_page_name = ''
            if page.get('ancestors') and len(page.get('ancestors')) > 0:
                parent_page_name = page.get('ancestors')[-1].get('title', '')

            # Extract the published date from the history field in ISO 8601 format
            published_date = ''
            if page.get('history'):
                published_date = page.get('history').get('createdDate', '')

            new_record = [{
                'id': page.get('id', ''),
                'type': page.get('type', ''),
                'status': page.get('status', ''),
                'tiny_link': page.get('_links', {}).get('tinyui', ''),
                'title': page.get('title', ''),
                'parent_page_name': parent_page_name,
                'published_date': published_date,
            }]

            # Add new records to the DataFrame
            df = pd.concat([df, pd.DataFrame(new_record)], ignore_index=True)
        except Exception as e:
            print(f"An error occurred while adding a page to the DataFrame: {e}")

    return df


# Function index of the DataFrame
def set_index_of_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        print("Error: The argument must be a pandas DataFrame.")
        return None

    if 'id' not in df.columns:
        print("Error: 'id' column not found in the DataFrame.")
        return None

    try:
        df.set_index('id', inplace=True)
        return df
    except Exception as e:
        print(f"An error occurred while setting the index: {e}")
        return None

# Function to fetch by limit
def fetch_pages_by_limit(all_pages, start, limit):
    if not isinstance(all_pages, list):
        print("Error: 'all_pages' must be a list.")
        return None

    while True:
        response_data = fetch_pages(start, limit)
        if response_data:
            results = response_data.get('results')
            if results:
                all_pages.extend(results)
                start += limit
                if start >= response_data.get('size', 0):
                    break
            else:
                print("Warning: No results found in the response.")
                break
        else:
            print("Error: Failed to fetch pages.")
            return None

    return all_pages

def fetch_all_pages(all_pages, start, limit, max_chunk_size=200):
    if not isinstance(all_pages, list):
        print("Error: 'all_pages' must be a list.")
        return None

    print(f"\nStarting fetch_all_pages with:")
    print(f"Start: {start}")
    print(f"Limit: {limit}")
    print(f"Max chunk size: {max_chunk_size}")
    print(f"Current all_pages length: {len(all_pages)}")

    # Get the total number of pages in the space first for verification
    first_response = fetch_pages(0, 1)
    total_space_pages = 0
    if first_response:
        total_space_pages = first_response.get('size', 0)
        print(f"\n*** IMPORTANT: Total pages available in space {space_key}: {total_space_pages} ***")
        if total_space_pages < limit:
            print(f"Note: You requested {limit} pages but there are only {total_space_pages} pages available in this space.")
            print("The script will fetch all available pages.")

    # Calculate the total number of chunks to fetch based on the limit and max_chunk_size
    total_chunks = (limit + max_chunk_size - 1) // max_chunk_size
    print(f"Total chunks to fetch: {total_chunks}")

    # Initialize the tqdm progress bar with the smaller of limit or total pages
    pages_to_fetch = min(limit, total_space_pages) if total_space_pages > 0 else limit
    with tqdm(total=pages_to_fetch, desc="Fetching pages") as pbar:
        while True:
            chunk_size = min(limit, max_chunk_size)  # Determine the size of the next chunk
            print(f"\nFetching chunk of size: {chunk_size}")
            response_data = fetch_pages(start, chunk_size)
            
            if response_data:
                results = response_data.get('results')
                if results is not None:
                    print(f"Found {len(results)} results in this chunk")
                    all_pages.extend(results)
                    fetched_count = len(results)
                    pbar.update(fetched_count)
                    if fetched_count < chunk_size:
                        print("Breaking loop: fetched count less than chunk size (reached end of available pages)")
                        break
                    start += fetched_count
                    limit -= fetched_count
                    if limit <= 0:
                        print("Breaking loop: limit reached")
                        break
                else:
                    print("Warning: No results found in the response.")
                    print(f"Response data: {response_data}")
                    break
            else:
                print("Error: Failed to fetch pages.")
                return None
            
    print(f"\nFinal all_pages length: {len(all_pages)}")
    return all_pages


# Function to delete internal_only records
def delete_internal_only_records(df):
    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: The variable 'df' must be a pandas DataFrame.")
        return df
    
    # Add logging for the initial dataframe size
    initial_size = df.shape[0]
    print(f"Initial DataFrame size before filtering internal records: {initial_size}")
    
    # Loop through the DataFrame with a tqdm progress bar
    if 'is_internal' in df.columns:
        for page_id, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating is_internal status"):
            is_internal_page = fetch_labels(page_id)
            
            if is_internal_page is not None:
                df.loc[page_id, 'is_internal'] = is_internal_page
            else:
                print(f"Warning: Could not fetch labels for page ID {page_id}.")
    else:
        print("Error: 'is_internal' column not found in the DataFrame.")
        return df
    
    # Log how many internal pages were found
    internal_count = df[df['is_internal'] == True].shape[0]
    print(f"Found {internal_count} internal pages that will be filtered out")
    
    # Delete internal_only records
    df = df[df['is_internal'] != True]
    
    # Log final size
    final_size = df.shape[0]
    print(f"Final DataFrame size after filtering: {final_size}")
    print(f"Removed {initial_size - final_size} internal pages")

    return df


def add_content_to_dataframe(df):
    # Check if the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: The variable 'df' must be a pandas DataFrame.")
        return df

    # Count how many pages have content initially
    content_before = df['content'].notna().sum()
    print(f"Pages with content before processing: {content_before}")

    # Wrap the loop in tqdm for progress tracking
    for page_id, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating DataFrame with content"):
        print(f"\nProcessing page ID: {page_id}")  # Debug logging
        html_content = fetch_page_content(page_id)

        if html_content is not None:
            try:
                # Parse the HTML content
                soup = BeautifulSoup(html_content, "html.parser")  # Changed from lxml to html.parser

                # Remove script and style elements
                for element in soup(["script", "style"]):
                    element.decompose()

                # Get text and handle whitespace
                page_content = soup.get_text(separator=' ').strip()
                # Clean up excessive whitespace
                page_content = ' '.join(page_content.split())

                if page_content:
                    print(f"Content length: {len(page_content)} characters")  # Debug logging
                else:
                    print("Warning: Extracted content is empty")

                # Update the DataFrame with the extracted content
                df.at[page_id, 'content'] = page_content

            except Exception as e:
                print(f"Error processing HTML content for page ID {page_id}: {e}")
        else:
            print(f"Warning: Could not fetch content for page ID {page_id}.")

    # Count how many pages have content after processing
    content_after = df['content'].notna().sum()
    print(f"Pages with content after processing: {content_after}")
    print(f"Added content to {content_after - content_before} pages")

    return df


def save_dataframe_to_csv(df, filename):
    if not isinstance(df, pd.DataFrame):
        print("Error: The variable 'df' must be a pandas DataFrame.")
    else:
        try:
            df.to_csv(filename, index=True)
            print(f"Data successfully saved - {len(df)} records written to {filename}")
            
            # Additional information about the saved data
            print("\nSummary of saved data:")
            print(f"Total rows: {df.shape[0]}")
            print(f"Total columns: {df.shape[1]}")
            print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
            
            # Check for any missing content
            missing_content = df['content'].isna().sum()
            if missing_content > 0:
                print(f"Warning: {missing_content} pages ({missing_content/len(df)*100:.1f}%) have missing content")
                
        except Exception as e:
            print(f"An error occurred while saving the DataFrame to CSV: {e}")

def main():
    # Fetch pages on limit occurance
    all_pages = []
    start = 0
    limit = 2000
    csv_file = './conf_data.csv'
    
    print("\n======== STARTING CONFLUENCE SCRAPER ========")
    print(f"Confluence domain: {confluence_domain}")
    print(f"Space key: {space_key}")
    print(f"Environment variables loaded: {bool(confluence_domain and api_key)}")
    print(f"Target number of pages: {limit}")
    print("=============================================")
    
    print("\nStep 1: Fetching pages...")
    all_pages = fetch_all_pages(all_pages, start, limit)
    
    if all_pages is None:
        print("Error: Failed to fetch pages. Exiting...")
        return
        
    print(f"\nStep 2: Processing {len(all_pages)} fetched pages...")
    df = create_dataframe()
    df = add_all_pages_to_dataframe(df, all_pages)
    df = set_index_of_dataframe(df)
    
    print("\nStep 3: Filtering out internal-only pages...")
    df = delete_internal_only_records(df)
    
    print("\nStep 4: Adding content to DataFrame...")
    df = add_content_to_dataframe(df)
    
    print("\nStep 5: Saving data to CSV...")
    save_dataframe_to_csv(df, csv_file)
    
    print("\n======== CONFLUENCE SCRAPER COMPLETE ========")
    print(f"Started with request for {limit} pages")
    print(f"Found and processed {len(all_pages)} pages")
    print(f"Final dataset contains {len(df)} pages after filtering")
    print("==============================================")


if __name__ == "__main__":
    main()