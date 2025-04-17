from flask import Flask, render_template, request, jsonify, send_file
import praw
import json
import os
import re
from datetime import datetime
import time
from dotenv import load_dotenv
import io
from reddit_converter.reddit_search import search_reddit_directly, get_reddit_post_data
from reddit_converter.data_processor import clean_reddit_data, enhance_data_for_llm, convert_to_chatml

# Load environment variables
load_dotenv()

# Configure app
app = Flask(__name__)

# Reddit API configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize PRAW
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

# Import functions from your existing scripts
# [Copy relevant functions from 01.py and 02.py]
# Include: search_reddit_directly, get_reddit_post_data, 
# clean_text, filter_valid_comments, enhance_data_for_llm, convert_to_chatml

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    subreddit = request.form.get('subreddit', 'all')
    limit = int(request.form.get('limit', 5))
    
    # Cap the limit at 10 as per requirements
    limit = min(limit, 10)
    
    if not query:
        return jsonify({"error": "Search query is required"}), 400
    
    try:
        # Search Reddit
        submissions = search_reddit_directly(query, reddit, subreddit_name=subreddit, limit=limit)
        
        all_posts_data = []
        for submission in submissions:
            data = get_reddit_post_data(submission)
            if data:
                all_posts_data.append(data)
            time.sleep(0.5)  # Be polite to Reddit API
        
        if not all_posts_data:
            return jsonify({"error": "No posts found for your query"}), 404
        
        # Clean and enhance data
        cleaned_data = clean_reddit_data(all_posts_data)
        enhanced_data = enhance_data_for_llm(cleaned_data)
        
        # Convert to ChatML
        chatml_data = convert_to_chatml(enhanced_data)
        
        # Prepare file for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'\W+', '_', query)
        filename = f"reddit_chatml_{safe_query}_{subreddit}_{timestamp}.json"
        
        # Create in-memory file
        mem_file = io.BytesIO()
        mem_file.write(json.dumps(chatml_data, indent=2).encode('utf-8'))
        mem_file.seek(0)
        
        return send_file(
            mem_file,
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)