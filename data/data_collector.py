import os
import csv
import json
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
import tweepy
import praw
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging with more detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Environment Variable Validation === #
def validate_env_vars():
    """Check if required environment variables are set"""
    required_vars = {
        'TWITTER_BEARER_TOKEN': os.getenv("TWITTER_BEARER_TOKEN"),
        'REDDIT_CLIENT_ID': os.getenv("REDDIT_CLIENT_ID"),
        'REDDIT_CLIENT_SECRET': os.getenv("REDDIT_CLIENT_SECRET"),
        'REDDIT_USER_AGENT': os.getenv("REDDIT_USER_AGENT"),
        'NEWS_API_KEY': os.getenv("NEWS_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        return False
    return True

# === API Setup with Error Handling === #
def setup_apis():
    """Initialize API clients with proper error handling"""
    clients = {}
    
    # Twitter Setup
    twitter_token = os.getenv("TWITTER_BEARER_TOKEN")
    if twitter_token:
        try:
            clients['twitter'] = tweepy.Client(bearer_token=twitter_token, wait_on_rate_limit=True)
            # Test Twitter connection
            me = clients['twitter'].get_me()
            logger.info("Twitter API connection successful")
        except Exception as e:
            logger.error(f"Twitter API setup failed: {e}")
            clients['twitter'] = None
    else:
        logger.warning("Twitter Bearer Token not found")
        clients['twitter'] = None

    # Reddit Setup
    reddit_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_agent = os.getenv("REDDIT_USER_AGENT")
    
    if all([reddit_id, reddit_secret, reddit_agent]):
        try:
            clients['reddit'] = praw.Reddit(
                client_id=reddit_id,
                client_secret=reddit_secret,
                user_agent=reddit_agent
            )
            # Test Reddit connection
            clients['reddit'].user.me()
            logger.info("Reddit API connection successful")
        except Exception as e:
            logger.error(f"Reddit API setup failed: {e}")
            clients['reddit'] = None
    else:
        logger.warning("Reddit credentials not found")
        clients['reddit'] = None
    
    return clients

# Initialize API clients
api_clients = setup_apis()

# === Database Setup === #
DB_PATH = "processed/collected_data.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db_connection():
    """Get database connection with proper error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT PRIMARY KEY,
                text TEXT,
                created_at TEXT,
                username TEXT,
                language TEXT,
                location TEXT,
                retweet_count INTEGER,
                like_count INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id TEXT PRIMARY KEY,
                subreddit TEXT,
                title TEXT,
                score INTEGER,
                num_comments INTEGER,
                created_utc TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                source TEXT,
                author TEXT,
                title TEXT,
                description TEXT,
                published_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if conn:
            conn.close()
        return False

# === Twitter Collection === #
def collect_tweets(query: str, lang='en', max_results=100):
    """Collect tweets with improved error handling and logging"""
    if not api_clients.get('twitter'):
        logger.warning("Twitter client not available, skipping tweet collection")
        return 0
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cursor = conn.cursor()
        tweet_count = 0
        
        logger.info(f"Searching for tweets with query: '{query}'")
        
        # Use search_recent_tweets with proper pagination
        tweets = tweepy.Paginator(
            api_clients['twitter'].search_recent_tweets,
            query=query,
            tweet_fields=['created_at', 'lang', 'public_metrics', 'author_id'],
            user_fields=['username', 'location'],
            expansions='author_id',
            max_results=min(100, max_results)
        ).flatten(limit=max_results)

        # Get user data for username mapping
        users = {}
        for tweet in tweets:
            if hasattr(tweet, 'includes') and 'users' in tweet.includes:
                for user in tweet.includes['users']:
                    users[user.id] = user

        # Reset tweets iterator
        tweets = tweepy.Paginator(
            api_clients['twitter'].search_recent_tweets,
            query=query,
            tweet_fields=['created_at', 'lang', 'public_metrics', 'author_id'],
            user_fields=['username', 'location'],
            expansions='author_id',
            max_results=min(100, max_results)
        ).flatten(limit=max_results)

        for tweet in tweets:
            try:
                metrics = tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
                username = str(tweet.author_id)  # Fallback to author_id if username not available
                
                cursor.execute("""
                    INSERT OR IGNORE INTO tweets (id, text, created_at, username, language, location, retweet_count, like_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(tweet.id),
                    tweet.text,
                    tweet.created_at.isoformat() if tweet.created_at else '',
                    username,
                    tweet.lang if hasattr(tweet, 'lang') else lang,
                    '',  # Location not easily available in v2 API
                    metrics.get("retweet_count", 0),
                    metrics.get("like_count", 0)
                ))
                tweet_count += 1
            except Exception as e:
                logger.error(f"Error processing individual tweet: {e}")
                continue
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully collected {tweet_count} tweets")
        return tweet_count
        
    except Exception as e:
        logger.error(f"Error collecting tweets: {e}")
        if conn:
            conn.close()
        return 0

# === Reddit Collection === #
def collect_reddit(subreddit_name: str, keyword: str = '', limit: int = 50):
    """Collect Reddit posts with improved error handling and multiple sorting methods"""
    if not api_clients.get('reddit'):
        logger.warning("Reddit client not available, skipping Reddit collection")
        return 0
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cursor = conn.cursor()
        post_count = 0
        total_posts_checked = 0
        
        logger.info(f"Collecting from subreddit: r/{subreddit_name}, keyword: '{keyword}', limit: {limit}")
        
        subreddit = api_clients['reddit'].subreddit(subreddit_name)
        
        # Try multiple sorting methods to get more diverse results
        sorting_methods = [
            ('hot', subreddit.hot),
            ('new', subreddit.new),
            ('top', lambda **kwargs: subreddit.top(time_filter='week', **kwargs))
        ]
        
        posts_per_method = max(1, limit // len(sorting_methods))
        
        for method_name, method_func in sorting_methods:
            try:
                logger.info(f"Fetching {method_name} posts from r/{subreddit_name}")
                
                for post in method_func(limit=posts_per_method):
                    try:
                        total_posts_checked += 1
                        
                        # Log some details about each post for debugging
                        logger.debug(f"Checking post: '{post.title[:50]}...' (Score: {post.score})")
                        
                        # More flexible keyword matching
                        title_text = post.title.lower()
                        selftext = getattr(post, 'selftext', '').lower()
                        combined_text = f"{title_text} {selftext}"
                        
                        # Check keyword filter (if no keyword, accept all posts)
                        if keyword:
                            keywords = [k.strip().lower() for k in keyword.split(',')]
                            if not any(kw in combined_text for kw in keywords):
                                continue
                        
                        # Check if post already exists
                        cursor.execute("SELECT id FROM reddit_posts WHERE id = ?", (post.id,))
                        if cursor.fetchone():
                            logger.debug(f"Post {post.id} already exists, skipping")
                            continue
                        
                        cursor.execute("""
                            INSERT OR IGNORE INTO reddit_posts (id, subreddit, title, score, num_comments, created_utc)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            post.id,
                            subreddit.display_name,
                            post.title,
                            post.score,
                            post.num_comments,
                            datetime.utcfromtimestamp(post.created_utc).isoformat()
                        ))
                        post_count += 1
                        logger.debug(f"Added post: {post.title[:50]}...")
                        
                    except Exception as e:
                        logger.error(f"Error processing individual Reddit post: {e}")
                        continue
                        
                logger.info(f"Collected {post_count} posts so far using {method_name} method")
                
            except Exception as e:
                logger.error(f"Error with {method_name} method: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"Reddit collection summary:")
        logger.info(f"  - Total posts checked: {total_posts_checked}")
        logger.info(f"  - Posts matching criteria: {post_count}")
        logger.info(f"  - Keyword filter: '{keyword}' (empty = no filter)")
        
        if post_count == 0:
            logger.warning("No Reddit posts collected. Possible reasons:")
            logger.warning("  1. Keyword filter too restrictive")
            logger.warning("  2. All posts already exist in database")
            logger.warning("  3. Subreddit has no recent posts")
            logger.warning("  4. API rate limiting or connection issues")
        
        return post_count
        
    except Exception as e:
        logger.error(f"Error collecting Reddit posts: {e}")
        if conn:
            conn.close()
        return 0

# === NewsAPI Collection === #
def collect_news(query: str, from_date: Optional[str] = None, to_date: Optional[str] = None, page_size=100):
    """Collect news articles with improved error handling"""
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        logger.warning("News API key not available, skipping news collection")
        return 0
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cursor = conn.cursor()
        article_count = 0
        
        logger.info(f"Collecting news articles for query: '{query}'")
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'apiKey': news_api_key,
            'pageSize': min(page_size, 100),  # API limit is 100
            'page': 1,
            'sortBy': 'publishedAt'
        }
        
        seen_ids = set()
        
        while True:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"News API error: {data.get('message', 'Unknown error')}")
                break
            
            articles = data.get('articles', [])
            if not articles:
                break
            
            for article in articles:
                try:
                    article_id = article.get('url', '')
                    if not article_id or article_id in seen_ids:
                        continue
                    
                    seen_ids.add(article_id)
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO news_articles (id, source, author, title, description, published_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        article_id,
                        article.get('source', {}).get('name', '') if article.get('source') else '',
                        article.get('author', ''),
                        article.get('title', ''),
                        article.get('description', ''),
                        article.get('publishedAt', '')
                    ))
                    article_count += 1
                except Exception as e:
                    logger.error(f"Error processing individual article: {e}")
                    continue
            
            # Check if we should continue pagination
            if len(articles) < params['pageSize'] or article_count >= page_size:
                break
            
            params['page'] += 1
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully collected {article_count} news articles")
        return article_count
        
    except Exception as e:
        logger.error(f"Error collecting news articles: {e}")
        if conn:
            conn.close()
        return 0

# === Database Inspection Functions === #
def check_db_contents():
    """Check what's actually in the database"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Check tweets
        cursor.execute("SELECT COUNT(*) FROM tweets")
        tweet_count = cursor.fetchone()[0]
        logger.info(f"Database contains {tweet_count} tweets")
        
        # Check Reddit posts
        cursor.execute("SELECT COUNT(*) FROM reddit_posts")
        reddit_count = cursor.fetchone()[0]
        logger.info(f"Database contains {reddit_count} Reddit posts")
        
        # Check news articles
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        news_count = cursor.fetchone()[0]
        logger.info(f"Database contains {news_count} news articles")
        
        # Show sample data if available
        if tweet_count > 0:
            cursor.execute("SELECT text FROM tweets LIMIT 1")
            sample_tweet = cursor.fetchone()[0]
            logger.info(f"Sample tweet: {sample_tweet[:100]}...")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking database contents: {e}")
        if conn:
            conn.close()

# === File Upload === #
def process_file_upload(filepath: str):
    """Process uploaded files with better error handling"""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
    
    ext = Path(filepath).suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext == '.json':
            df = pd.read_json(filepath)
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame({'text': [line.strip() for line in lines if line.strip()]})
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"Successfully loaded file {filepath} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error processing file upload {filepath}: {e}")
        return None

# === Main Collection Function === #
def scheduled_collect():
    """Main collection function with better logging and error handling"""
    logger.info("Starting scheduled data collection...")
    
    # Validate environment variables first
    if not validate_env_vars():
        logger.warning("Some environment variables are missing. Some data sources may not work.")
    
    total_collected = 0
    
    # Collect tweets
    try:
        tweet_count = collect_tweets("#AI OR artificial intelligence", lang='en', max_results=50)
        total_collected += tweet_count
    except Exception as e:
        logger.error(f"Tweet collection failed: {e}")
    
    # Collect Reddit posts
    try:
        # Try multiple subreddits and broader keywords
        subreddits_to_check = [
            ("technology", "AI,artificial intelligence,machine learning,automation"),
            ("MachineLearning", ""),  # No keyword filter for ML subreddit
            ("artificial", ""),       # Dedicated AI subreddit
            ("programming", "AI,artificial intelligence,machine learning"),
        ]
        
        reddit_count = 0
        for subreddit, keywords in subreddits_to_check:
            try:
                count = collect_reddit(subreddit, keyword=keywords, limit=15)
                reddit_count += count
                if count > 0:
                    logger.info(f"Collected {count} posts from r/{subreddit}")
            except Exception as e:
                logger.error(f"Failed to collect from r/{subreddit}: {e}")
                continue
        
        total_collected += reddit_count
        logger.info(f"Total Reddit posts collected: {reddit_count}")
        
    except Exception as e:
        logger.error(f"Reddit collection failed: {e}")
    
    # Collect news
    try:
        # Get articles from the last 7 days
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news_count = collect_news("artificial intelligence", from_date=from_date, page_size=50)
        total_collected += news_count
    except Exception as e:
        logger.error(f"News collection failed: {e}")
    
    logger.info(f"Collection completed. Total items collected: {total_collected}")
    
    # Check database contents
    check_db_contents()

if __name__ == "__main__":
    logger.info("Initializing data collection script...")
    
    # Initialize database
    if not init_db():
        logger.error("Database initialization failed. Exiting.")
        exit(1)
    
    # Run collection
    scheduled_collect()
    
    logger.info("Script execution completed.")