import praw
import json
import re
import os
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get credentials from environment variables
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Check if environment variables are set
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT:
    print("ERROR: Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in your .env file.")
    exit()

# Initialize PRAW (Reddit API Wrapper)
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        # Add username/password if you need authenticated actions, otherwise read-only is fine
        # username=os.getenv("REDDIT_USERNAME"),
        # password=os.getenv("REDDIT_PASSWORD"),
    )
    # Test read-only status
    print(f"PRAW initialized. Read-only: {reddit.read_only}")
except Exception as e:
    print(f"Error initializing PRAW: {e}")
    exit()

def search_reddit_directly(query, praw_instance, subreddit_name="all", limit=15, sort='relevance'):
    """Searches Reddit directly using PRAW."""
    print(f"Searching Reddit ({subreddit_name}) for: '{query}' (sort: {sort}, limit: {limit})")
    try:
        if subreddit_name.lower() == "all":
            subreddit = praw_instance.subreddit("all")
        else:
            subreddit = praw_instance.subreddit(subreddit_name)

        # Perform the search
        search_results = subreddit.search(query, sort=sort, limit=limit)

        # PRAW's search returns a generator. Convert to list or iterate.
        submissions = list(search_results) # Materialize the generator to get a count

        print(f"Found {len(submissions)} potential posts via PRAW search.")
        return submissions
    except praw.exceptions.PRAWException as e:
        print(f"PRAW Error during search: {e}")
        return []
    except Exception as e:
        print(f"General Error during search: {e}")
        return []

def get_reddit_post_data(submission):
    """Fetches post details and comments from a PRAW Submission object."""
    post_url = submission.permalink # For logging
    print(f"Fetching comments for: {post_url}")
    try:
        # Ensure the submission object has full data (sometimes needed if obtained from search)
        # submission.load() # Often not needed if iterating directly from search results, but can ensure all attributes are present

        post_data = {
            "post_title": getattr(submission, 'title', 'N/A'),
            "post_author": str(getattr(submission, 'author', 'N/A')),
            "post_score": getattr(submission, 'score', 0),
            "post_url": submission.permalink,
            "post_body": getattr(submission, 'selftext', ''),
            "post_id": submission.id,
            "subreddit": str(getattr(submission, 'subreddit', 'N/A')),
            "created_utc": getattr(submission, 'created_utc', 0.0),
            "num_comments_reported": getattr(submission, 'num_comments', 0),
            "comments": []
        }

        retries = 3
        for i in range(retries):
            try:
                # Fetch comments - replace_more fetches all top-level comments
                submission.comments.replace_more(limit=0) # Fetch all top-level comments
                break # Success
            except praw.exceptions.ClientException as e:
                 # Handle potential issues like "Too many requests" during replace_more
                if "received 404 HTTP response" in str(e).lower():
                     print(f"-> Warning: Received 404 fetching comments for {post_url}. Post might be deleted or restricted.")
                     return post_data # Return post data without comments
                if i < retries - 1:
                    wait_time = 2**(i+1) # Exponential backoff
                    print(f"-> Rate limit or error during replace_more for {post_url}. Retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"-> Failed to fetch comments for {post_url} after {retries} retries: {e}")
                    # Decide if you want to return partial data or None
                    return post_data # Return post data without comments

        comment_count = 0
        # Use list(submission.comments) to process all fetched top-level comments
        for comment in submission.comments.list():
            # Skip MoreComments objects if any remain after replace_more
            if isinstance(comment, praw.models.MoreComments):
                continue

            post_data["comments"].append({
                "comment_id": comment.id,
                "comment_author": str(getattr(comment, 'author', 'N/A')),
                "comment_body": getattr(comment, 'body', ''),
                "comment_score": getattr(comment, 'score', 0),
                "comment_permalink": getattr(comment, 'permalink', ''),
                "created_utc": getattr(comment, 'created_utc', 0.0),
            })
            comment_count += 1
            # Optional: Add recursion here if you want nested replies
            # Be mindful of API limits and complexity

        print(f"-> Fetched {comment_count} top-level comments for post '{post_data['post_title']}'.")
        return post_data

    except praw.exceptions.PRAWException as e:
        # Handle cases where the submission might be deleted or inaccessible
        if "received 404 HTTP response" in str(e).lower() or \
           "received 403 HTTP response" in str(e).lower():
             print(f"-> PRAW Error: Post {post_url} seems inaccessible (deleted, private, etc.). Skipping. ({e})")
        else:
            print(f"-> PRAW Error processing {post_url}: {e}")
    except Exception as e:
        print(f"-> General Error processing {post_url}: {e}")
    return None # Return None if processing failed critically

# --- Main Execution ---
if __name__ == "__main__":
    user_query = input("Enter your search query for Reddit: ")
    # Optional: Ask for specific subreddit or use 'all'
    subreddit_target = input("Enter subreddit to search (e.g., MachineLearning, leave blank for 'all'): ")
    if not subreddit_target:
        subreddit_target = "all"

    # Optional: Ask for result limit and sort order
    try:
        limit = int(input("How many posts to retrieve (e.g., 10)? "))
    except ValueError:
        limit = 10 # Default limit
    sort_order = input("Sort by (relevance, top, new, comments - default relevance)? ").lower()
    if sort_order not in ['relevance', 'top', 'new', 'comments']:
        sort_order = 'relevance'

    if not user_query:
        print("Search query cannot be empty.")
    else:
        # Search Reddit directly using PRAW
        found_submissions = search_reddit_directly(
            user_query,
            reddit,
            subreddit_name=subreddit_target,
            limit=limit,
            sort=sort_order
            )

        all_posts_data = []
        if found_submissions:
            print(f"\nProcessing {len(found_submissions)} found submissions...")
            for submission in found_submissions:
                # Pass the submission object directly
                data = get_reddit_post_data(submission)
                if data:
                    all_posts_data.append(data)
                time.sleep(1) # Add a small delay between processing posts to be polite to API
        else:
            print("No relevant posts found via PRAW search.")

        if all_posts_data:
            # Ask for custom output filename
            custom_filename = input("Enter the output filename (without extension): ")
            
            if not custom_filename:
                # Create a default filename based on the query and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = re.sub(r'\W+', '_', user_query) # Make query filename-safe
                filename = f"reddit_praw_report_{safe_query}_{subreddit_target}_{timestamp}.json"
            else:
                # Use the custom filename
                filename = f"{custom_filename}.json"

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(all_posts_data, f, ensure_ascii=False, indent=4)
                print(f"\nSuccessfully saved data for {len(all_posts_data)} posts to {filename}")
            except IOError as e:
                print(f"Error writing JSON file: {e}")
        else:
            print("\nNo data collected from Reddit.")
    
    # Print True when execution completes
    print(True)