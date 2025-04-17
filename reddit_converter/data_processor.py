import json
import pandas as pd
from datetime import datetime
import re
from typing import List, Dict, Any, Optional, Union
import os
import argparse
from dotenv import load_dotenv
import tiktoken
import hashlib

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables with defaults
MAX_TOKENS_PER_MESSAGE = int(os.getenv("MAX_TOKENS_PER_MESSAGE", "4000"))
DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant analyzing Reddit discussions.")
SENTIMENT_ANALYSIS = os.getenv("SENTIMENT_ANALYSIS", "False").lower() in ('true', '1', 'yes')

def load_reddit_data(file_path: str) -> List[Dict[Any, Any]]:
    """
    Load Reddit data from a JSON file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} Reddit posts")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def convert_timestamp(timestamp: float) -> str:
    """
    Convert Unix timestamp to human-readable date
    """
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ""

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    """
    if not text or text.lower() in ["deleted", "removed", "overwritten", ""]:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs for cleaner text
    text = re.sub(r'http\S+', '[URL]', text)
    
    # Remove Reddit formatting artifacts
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*|\*|~~|__|\^\^', '', text)
    
    # Remove quote markers
    text = re.sub(r'^\s*>\s*', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&amp;|&lt;|&gt;|&nbsp;', ' ', text)
    
    return text

def basic_sentiment_analysis(text: str) -> str:
    """
    Perform a very basic sentiment analysis
    """
    if not text:
        return "neutral"
        
    # Simple keyword-based approach
    positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'best', 'nice', 'thanks', 
                      'helpful', 'wonderful', 'amazing', 'happy', 'like', 'agree', 'perfect']
    negative_words = ['bad', 'awful', 'terrible', 'hate', 'worst', 'poor', 'horrible', 'sucks', 
                      'wrong', 'disappointed', 'stupid', 'annoying', 'dislike', 'disagree', 'terrible']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if f" {word} " in f" {text_lower} ")
    negative_count = sum(1 for word in negative_words if f" {word} " in f" {text_lower} ")
    
    if positive_count > negative_count * 2:
        return "positive"
    elif negative_count > positive_count * 2:
        return "negative"
    elif positive_count > negative_count:
        return "slightly_positive"
    elif negative_count > positive_count:
        return "slightly_negative"
    else:
        return "neutral"

def filter_valid_comments(comments: List[Dict[Any, Any]], include_sentiment: bool = False) -> List[Dict[Any, Any]]:
    """
    Filter out deleted/overwritten comments and normalize
    """
    if not comments:
        return []
        
    valid_comments = []
    for comment in comments:
        # Skip deleted/overwritten comments
        body = comment.get("comment_body", "")
        if not body or body.lower() in ["deleted", "removed", "overwritten"]:
            continue
            
        # Clean the comment
        cleaned_body = clean_text(body)
        
        cleaned_comment = {
            "id": comment.get("comment_id", ""),
            "author": comment.get("comment_author", ""),
            "body": cleaned_body,
            "score": comment.get("comment_score", 0),
            "permalink": comment.get("comment_permalink", ""),
            "created_at": convert_timestamp(comment.get("created_utc", 0))
        }
        
        # Add sentiment analysis if requested
        if include_sentiment and cleaned_body:
            cleaned_comment["sentiment"] = basic_sentiment_analysis(cleaned_body)
            
        if cleaned_body:  # Only add if body is not empty
            valid_comments.append(cleaned_comment)
            
    return valid_comments

def extract_entities(text: str) -> List[str]:
    """
    Extract potential named entities (simplistic approach)
    """
    # This is a basic approach - a real NLP library would be better
    # Look for capitalized words that aren't at the start of sentences
    entities = set()
    
    # First pass: find capitalized word sequences
    matches = re.finditer(r'(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
    for match in matches:
        entity = match.group(1)
        if len(entity) > 2 and entity.lower() not in ['the', 'and', 'but', 'for', 'or', 'yet', 'so', 'i', 'you']:
            entities.add(entity)
            
    return list(entities)

def clean_reddit_data(data: List[Dict[Any, Any]], include_sentiment: bool = False) -> List[Dict[Any, Any]]:
    """
    Clean the Reddit data for better processing with Llama3
    """
    cleaned_data = []
    
    for post in data:
        # Clean post body
        post_body = clean_text(post.get("post_body", ""))
        post_title = clean_text(post.get("post_title", ""))
        
        # Clean and filter comments
        comments = filter_valid_comments(post.get("comments", []), include_sentiment)
        
        # Skip if post has no title and no valid comments
        if not post_title and not comments:
            continue
            
        # Create cleaned post structure
        cleaned_post = {
            "id": post.get("post_id", ""),
            "title": post_title,
            "author": post.get("post_author", ""),
            "body": post_body,
            "score": post.get("post_score", 0),
            "url": post.get("post_url", ""),
            "subreddit": post.get("subreddit", ""),
            "created_at": convert_timestamp(post.get("created_utc", 0)),
            "comment_count": len(comments),
            "comments": comments
        }
        
        # Add sentiment analysis to post if requested
        if include_sentiment:
            combined_text = f"{post_title} {post_body}"
            if combined_text.strip():
                cleaned_post["sentiment"] = basic_sentiment_analysis(combined_text)
            else:
                cleaned_post["sentiment"] = "neutral"
        
        cleaned_data.append(cleaned_post)
        
    return cleaned_data

def enhance_data_for_llm(data: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
    """
    Add features that will help Llama3 process the data more effectively
    """
    enhanced_data = []
    
    for post in data:
        # Create a condensed text representation for the post
        post_content = f"{post['title']} {post['body']}"
        
        # Extract main topics/keywords (improved approach)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', post_content.lower())
        word_freq = {}
        for word in words:
            if word not in ['the', 'and', 'for', 'that', 'this', 'with', 'you', 'not', 'are', 'was', 'were', 'have', 'has', 'will', 'they', 'their', 'from', 'what', 'when', 'where', 'which', 'them', 'then', 'than', 'some', 'your']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:7]
        keywords = [k[0] for k in keywords]
        
        # Extract potential named entities
        entities = extract_entities(post_content)[:5]  # Limit to top 5
        
        # Add enhanced features
        enhanced_post = post.copy()
        enhanced_post["keywords"] = keywords
        enhanced_post["entities"] = entities
        
        # Add reading difficulty estimate (basic Flesch-Kincaid approximation)
        if post_content:
            sentences = len(re.split(r'[.!?]+', post_content))
            words = len(re.findall(r'\b\w+\b', post_content))
            if sentences > 0 and words > 0:
                avg_sentence_length = words / sentences
                if avg_sentence_length > 20:
                    difficulty = "complex"
                elif avg_sentence_length > 12:
                    difficulty = "moderate"
                else:
                    difficulty = "simple"
                enhanced_post["reading_complexity"] = difficulty
        
        # Add a post summary field (for Llama3 to use)
        if post['body']:
            summary = f"Post about {', '.join(keywords[:3])} in r/{post['subreddit']} with {post['comment_count']} comments"
        else:
            summary = f"Title-only post about {', '.join(keywords[:3])} in r/{post['subreddit']}"
            
        enhanced_post["summary"] = summary
        
        enhanced_data.append(enhanced_post)
        
    return enhanced_data

def estimate_token_count(text: str) -> int:
    """
    Estimate token count using tiktoken (if available) or a simpler approximation
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")  # encoding used by llama models
        return len(enc.encode(text))
    except (ImportError, ModuleNotFoundError):
        # Fall back to simple approximation - average English word is about 4.7 chars
        return len(text) // 4

def convert_to_chatml(data: List[Dict[Any, Any]], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert the Reddit data to ChatML format for Llama 3
    
    Basic ChatML format:
    {
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"}
        ]
    }
    """
    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        
    chatml_data = []
    
    for post in data:
        # Create a unique conversation ID based on post ID
        conversation_id = hashlib.md5(post["id"].encode()).hexdigest()
        
        # Construct the main post content
        post_content = f"Title: {post['title']}\n\n"
        if post["body"]:
            post_content += f"Post content: {post['body']}\n\n"
        post_content += f"Posted in r/{post['subreddit']} by u/{post['author']} on {post['created_at']}"
        
        # Add keywords and entities if available
        if post.get("keywords"):
            post_content += f"\n\nKeywords: {', '.join(post['keywords'])}"
        if post.get("entities"):
            post_content += f"\n\nEntities mentioned: {', '.join(post['entities'])}"
            
        # Construct conversation
        conversation = {
            "id": conversation_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": post_content}
            ]
        }
        
        # Add comments as conversation turns
        if post["comments"]:
            # Estimate token count to avoid exceeding model context
            token_count = estimate_token_count(post_content) + estimate_token_count(system_prompt)
            added_comments = 0
            
            for comment in post["comments"]:
                comment_text = f"Comment by u/{comment['author']} (score: {comment['score']}):\n{comment['body']}"
                comment_tokens = estimate_token_count(comment_text)
                
                # Check if adding this comment would exceed token limit
                if token_count + comment_tokens > MAX_TOKENS_PER_MESSAGE:
                    break
                    
                # Add comment as a message in conversation
                conversation["messages"].append({"role": "user", "content": comment_text})
                
                # Add placeholder for assistant response
                placeholder_response = "I understand. Let me analyze this comment."
                conversation["messages"].append({"role": "assistant", "content": placeholder_response})
                
                token_count += comment_tokens + estimate_token_count(placeholder_response)
                added_comments += 1
                
                # Limit to reasonable number of turns
                if added_comments >= 5:
                    break
                    
            # Add final prompt to summarize/analyze the thread
            final_prompt = f"Please analyze this Reddit thread from r/{post['subreddit']} about '{post['title']}'. What are the key points, main opinions, and interesting insights from this discussion?"
            conversation["messages"].append({"role": "user", "content": final_prompt})
        
        chatml_data.append(conversation)
        
    return {"conversations": chatml_data}

def write_cleaned_data(data: List[Dict[Any, Any]], output_path: str, format: str = "json") -> None:
    """
    Write cleaned data to a new file in specified format
    """
    try:
        if format == "chatml":
            # Convert to ChatML format
            chatml_data = convert_to_chatml(data)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chatml_data, f, ensure_ascii=False, indent=2)
        else:
            # Standard JSON format
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        print(f"Successfully wrote cleaned data to {output_path}")
    except Exception as e:
        print(f"Error writing data: {e}")

def write_summary_report(data: List[Dict[Any, Any]], output_path: str) -> None:
    """
    Write a summary report with statistics and insights
    """
    total_posts = len(data)
    total_comments = sum(post["comment_count"] for post in data)
    avg_comments = total_comments / total_posts if total_posts else 0
    
    subreddits = {}
    all_keywords = {}
    
    for post in data:
        # Count subreddits
        subreddit = post.get("subreddit", "unknown")
        subreddits[subreddit] = subreddits.get(subreddit, 0) + 1
        
        # Count keywords
        for keyword in post.get("keywords", []):
            all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
    
    # Sort by frequency
    top_subreddits = sorted(subreddits.items(), key=lambda x: x[1], reverse=True)
    top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Calculate sentiment distributions if available
    sentiment_stats = {}
    if any("sentiment" in post for post in data):
        for post in data:
            if "sentiment" in post:
                sentiment = post["sentiment"]
                sentiment_stats[sentiment] = sentiment_stats.get(sentiment, 0) + 1
    
    # Generate report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Reddit Data Summary Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n")
        f.write(f"- Total posts analyzed: {total_posts}\n")
        f.write(f"- Total comments: {total_comments}\n")
        f.write(f"- Average comments per post: {avg_comments:.1f}\n\n")
        
        f.write("## Subreddit Distribution\n")
        for subreddit, count in top_subreddits[:10]:  # Show top 10
            f.write(f"- r/{subreddit}: {count} posts ({count/total_posts*100:.1f}%)\n")
        
        f.write("\n## Top Keywords\n")
        for keyword, count in top_keywords[:15]:  # Show top 15
            f.write(f"- {keyword}: {count} occurrences\n")
            
        if sentiment_stats:
            f.write("\n## Sentiment Analysis\n")
            total_sentiment = sum(sentiment_stats.values())
            for sentiment, count in sorted(sentiment_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {sentiment}: {count} posts ({count/total_sentiment*100:.1f}%)\n")
        
        # Add sample post titles
        f.write("\n## Sample Post Titles\n")
        for i, post in enumerate(data[:10], 1):  # Show first 10 post titles
            f.write(f"{i}. {post['title']}\n")
            
    print(f"Summary report written to {output_path}")

def run_cleaning_pipeline(
    input_file: str, 
    output_file: str,
    output_format: str = "json",
    include_sentiment: bool = False,
    generate_report: bool = False
) -> None:
    """
    Run the complete cleaning pipeline
    """
    # Load the data
    data = load_reddit_data(input_file)
    if not data:
        print("No data to process. Exiting.")
        return
        
    # Clean the data
    print("Cleaning data...")
    cleaned_data = clean_reddit_data(data, include_sentiment=include_sentiment)
    print(f"Cleaned {len(cleaned_data)} posts")
    
    # Enhance for LLM
    print("Enhancing data for Llama3...")
    enhanced_data = enhance_data_for_llm(cleaned_data)
    
    # Write the cleaned data
    write_cleaned_data(enhanced_data, output_file, format=output_format)
    
    # Generate summary report if requested
    if generate_report:
        report_filename = os.path.splitext(output_file)[0] + "_report.md"
        write_summary_report(enhanced_data, report_filename)
    
    # Print statistics
    total_comments = sum(post["comment_count"] for post in enhanced_data)
    print(f"\nCleaning Summary:")
    print(f"- Total posts: {len(enhanced_data)}")
    print(f"- Total comments: {total_comments}")
    print(f"- Average comments per post: {total_comments/len(enhanced_data) if enhanced_data else 0:.1f}")
    
    # Print sample post to verify
    if enhanced_data:
        print("\nSample cleaned post:")
        sample = enhanced_data[0]
        print(f"Title: {sample['title']}")
        print(f"Keywords: {', '.join(sample['keywords'])}")
        print(f"Summary: {sample['summary']}")
        if sample['comments']:
            print(f"Sample comment: {sample['comments'][0]['body'][:100]}...")
    
    # Indicate successful completion
    print(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean and enhance Reddit JSON data')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('--output_file', help='Output JSON file path')
    parser.add_argument('--format', choices=['json', 'chatml'], default='json', 
                        help='Output format (default: json)')
    parser.add_argument('--sentiment', action='store_true', 
                        help='Include basic sentiment analysis')
    parser.add_argument('--report', action='store_true', 
                        help='Generate a summary report in markdown format')
    parser.add_argument('--system_prompt', type=str, 
                        help='Custom system prompt for ChatML format')
    parser.add_argument('--max_tokens', type=int, 
                        help='Maximum tokens per ChatML conversation')
    
    args = parser.parse_args()
    
    # Get input/output filenames
    input_file = args.input_file
    if args.output_file:
        output_file = args.output_file
    else:
        # Generate appropriate filename based on format
        basename = os.path.basename(input_file)
        if args.format == 'chatml':
            output_file = f"chatml_{basename}"
        else:
            output_file = f"cleaned_{basename}"
    
    # Override environment variables if provided as arguments
    if args.system_prompt:
        os.environ["DEFAULT_SYSTEM_PROMPT"] = args.system_prompt
    if args.max_tokens:
        os.environ["MAX_TOKENS_PER_MESSAGE"] = str(args.max_tokens)
    
    # Run the pipeline
    run_cleaning_pipeline(
        input_file, 
        output_file,
        output_format=args.format,
        include_sentiment=args.sentiment or SENTIMENT_ANALYSIS,
        generate_report=args.report
    )