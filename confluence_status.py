import requests
from requests.auth import HTTPBasicAuth
import sys
import os

# -------------------------------
# Configuration
# -------------------------------
CONFLUENCE_DOMAIN = "pickme.atlassian.net"  # e.g. "pickme.atlassian.net"
EMAIL = "haresha.perera@pickme.lk"          # Atlassian/Confluence user
API_TOKEN = os.getenv("CONF_API_KEY")  # from id.atlassian.com/manage/api-tokens
SPACE_KEY = "BDDS"                          # e.g. "ENG", "DOC", "DSA"

BASE_URL = f"https://{CONFLUENCE_DOMAIN}/wiki/rest/api"
AUTH = HTTPBasicAuth(EMAIL, API_TOKEN)

# -------------------------------
# 1) Get Space Info (to find homepage)
# -------------------------------
def get_space_homepage_id(space_key):
    """
    Returns the root page ID for the given space key.
    """
    # First, get the space info
    url = f"{BASE_URL}/space"
    params = {"spaceKey": space_key}
    resp = requests.get(url, auth=AUTH, params=params)
    
    if resp.status_code != 200:
        print(f"[ERROR] Cannot retrieve space info for {space_key}. HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)
    
    data = resp.json()
    results = data.get("results", [])
    
    if not results:
        print(f"[ERROR] Space '{space_key}' not found or you don't have access to it.")
        print("Please verify the space key and your access permissions.")
        sys.exit(1)
    
    # Then, get the root page (the first page found in that space)
    url = f"{BASE_URL}/content"
    params = {
        "spaceKey": space_key,
        "type": "page",
        "start": 0,
        "limit": 1
    }
    resp = requests.get(url, auth=AUTH, params=params)
    
    if resp.status_code != 200:
        print(f"[ERROR] Cannot retrieve root page for space {space_key}. HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)
    
    data = resp.json()
    results = data.get("results", [])
    
    if not results:
        print(f"[ERROR] No pages found in space '{space_key}'")
        sys.exit(1)
    
    root_page = results[0]
    return root_page["id"]

# -------------------------------
# 2) Get Children of a Page
# -------------------------------
def get_page_children(page_id):
    """
    Returns a list of child pages (id, title) for a given page_id.
    Uses expand=children.page to fetch direct children.
    """
    url = f"{BASE_URL}/content/{page_id}"
    params = {"expand": "children.page"}
    resp = requests.get(url, auth=AUTH, params=params)
    
    if resp.status_code != 200:
        print(f"[ERROR] Can't fetch children for page {page_id}. HTTP {resp.status_code}")
        return []
    
    data = resp.json()
    children_data = data.get("children", {}).get("page", {}).get("results", [])
    
    children = []
    for c in children_data:
        child_id = c["id"]
        child_title = c["title"]
        children.append((child_id, child_title))
    
    return children

# -------------------------------
# 3) Recursively Build Tree Lines
# -------------------------------
def build_page_tree_lines(page_id, page_title, indent=0):
    """
    Recursively gathers lines representing the hierarchy of pages.
    Returns a tuple: (list_of_lines, total_count_in_subtree).
    """
    prefix = " " * (indent * 4)  # 4 spaces per indent level
    line = f"{prefix}- {page_title} (ID: {page_id})"
    lines = [line]  # start with this page
    subtree_count = 1  # this page counts as 1

    children = get_page_children(page_id)
    for (child_id, child_title) in children:
        child_lines, child_count = build_page_tree_lines(child_id, child_title, indent + 1)
        lines.extend(child_lines)
        subtree_count += child_count
    
    return lines, subtree_count

# -------------------------------
# Main
# -------------------------------
def main():
    # Step A: Find the homepage ID for the space
    homepage_id = get_space_homepage_id(SPACE_KEY)
    if not homepage_id:
        print(f"[ERROR] Could not find homepage for space key: {SPACE_KEY}")
        return
    
    # Step B: Retrieve homepage title
    url = f"{BASE_URL}/content/{homepage_id}"
    resp = requests.get(url, auth=AUTH)
    if resp.status_code == 200:
        homepage_data = resp.json()
        homepage_title = homepage_data["title"]
    else:
        homepage_title = f"Homepage({homepage_id})"
    
    # Step C: Build the page tree lines
    lines, total_count = build_page_tree_lines(homepage_id, homepage_title, indent=0)
    
    # Step D: Write to conf_page_tree.txt
    output_file_path = os.path.join(os.getcwd(), "conf_page_tree.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    
    # Print a summary to console (optional)
    print(f"[INFO] Wrote page tree hierarchy to {output_file_path}")
    print(f"[INFO] Total pages in the '{SPACE_KEY}' hierarchy: {total_count}")

if __name__ == "__main__":
    main()
