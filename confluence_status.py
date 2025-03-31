import requests
from requests.auth import HTTPBasicAuth
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Configuration
# -------------------------------
CONFLUENCE_DOMAIN = "pickme.atlassian.net"  # e.g. "pickme.atlassian.net"
EMAIL = "haresha.perera@pickme.lk"          # Atlassian/Confluence user
API_TOKEN = os.getenv("CONF_API_KEY")        # from id.atlassian.com/manage/api-tokens
SPACE_KEY = "BDDS"                             # e.g. "ENG", "DOC", "DSA"
BASE_URL = f"https://{CONFLUENCE_DOMAIN}/wiki/rest/api"
AUTH = HTTPBasicAuth(EMAIL, API_TOKEN)

# -------------------------------
# 1) Get Space Info (to find homepage)
# -------------------------------
def get_space_homepage_id(space_key):
    """
    Returns the homepage ID for the given space key using the homepage expansion.
    """
    url = f"{BASE_URL}/space/{space_key}?expand=homepage"
    resp = requests.get(url, auth=AUTH)
    if resp.status_code != 200:
        print(f"[ERROR] Cannot retrieve space info for {space_key}. HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)
    data = resp.json()
    homepage = data.get("homepage")
    if not homepage:
        print(f"[ERROR] No homepage found for space '{space_key}'")
        sys.exit(1)
    return homepage["id"]

# -------------------------------
# 2) Get Children of a Page using the /child/page endpoint
# -------------------------------
def get_page_children(page_id):
    """
    Returns a list of child pages (id, title) for a given page_id.
    Handles pagination to ensure all children are fetched.
    """
    children = []
    limit = 25
    start = 0
    while True:
        url = f"{BASE_URL}/content/{page_id}/child/page"
        params = {"limit": limit, "start": start}
        resp = requests.get(url, auth=AUTH, params=params)
        if resp.status_code != 200:
            print(f"[ERROR] Can't fetch children for page {page_id}. HTTP {resp.status_code}")
            break
        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        children.extend([(child["id"], child["title"]) for child in batch])
        if len(batch) < limit:
            break
        start += limit
    return children

# -------------------------------
# 3) Recursively Build Tree Lines and Print in Real Time
# -------------------------------
def build_page_tree_lines(page_id, page_title, indent=0, visited=None):
    """
    Recursively builds a list of page tree lines, printing each page immediately.
    Uses a visited set to avoid cycles.
    Returns a tuple: (list_of_lines, total_count_in_subtree).
    """
    if visited is None:
        visited = set()
    if page_id in visited:
        return [], 0
    visited.add(page_id)
    
    prefix = " " * (indent * 4)  # 4 spaces per indent level
    line = f"{prefix}- {page_title} (ID: {page_id})"
    print(line, flush=True)  # Print immediately
    lines = [line]
    subtree_count = 1

    children = get_page_children(page_id)
    for (child_id, child_title) in children:
        child_lines, child_count = build_page_tree_lines(child_id, child_title, indent + 1, visited)
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
    homepage_title = resp.json().get("title", f"Homepage({homepage_id})") if resp.status_code == 200 else f"Homepage({homepage_id})"
    
    print("Building page tree:")
    # Step C: Build the page tree lines while printing in real time
    lines, total_count = build_page_tree_lines(homepage_id, homepage_title, indent=0)
    
    # Step D: Write the entire page tree to conf_page_tree.txt once done
    output_file_path = os.path.join(os.getcwd(), "conf_page_tree.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    
    print(f"[INFO] Wrote page tree hierarchy to {output_file_path}")
    print(f"[INFO] Total pages in the '{SPACE_KEY}' hierarchy: {total_count}")

if __name__ == "__main__":
    main()
