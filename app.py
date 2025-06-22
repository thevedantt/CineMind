from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import sqlite3
from datetime import datetime
import json

# Configure Gemini API Key
genai.configure(api_key="AIzaSyDOFTjxJN62XqAwexqW4MhGOkRcg96bzrI")

app = Flask(__name__)

# Database initialization
def init_db():
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommended_movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            type TEXT,
            release_year INTEGER,
            director TEXT,
            cast TEXT,
            listed_in TEXT,
            description TEXT,
            query_used TEXT,
            recommendation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            watched BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (movie_id) REFERENCES recommended_movies (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER,
            watched_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rating INTEGER,
            notes TEXT,
            FOREIGN KEY (movie_id) REFERENCES recommended_movies (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load data
df = pd.read_csv("netflix_titles.csv")

# Create textual representation for each movie
def create_text(row):
    return f"""Type: {row['type']},
Title: {row['title']},
Director: {row['director']},
Cast: {row['cast']},
Released: {row['release_year']},
Genres: {row['listed_in']},
Description: {row['description']}"""

df["textual_representation"] = df.apply(create_text, axis=1)
texts = df["textual_representation"].tolist()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(texts)

# Database helper functions
def save_recommended_movies(movies, query):
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    movie_ids = []
    for movie in movies:
        cursor.execute('''
            INSERT INTO recommended_movies 
            (title, type, release_year, director, cast, listed_in, description, query_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            movie['title'],
            movie['type'],
            movie['release_year'],
            movie['director'],
            movie['cast'],
            movie['listed_in'],
            movie['description'],
            query
        ))
        movie_ids.append(cursor.lastrowid)
    
    conn.commit()
    conn.close()
    return movie_ids

def get_favorites():
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT rm.*, f.id as fav_id, f.watched, f.added_date
        FROM recommended_movies rm
        JOIN favorites f ON rm.id = f.movie_id
        ORDER BY f.added_date DESC
    ''')
    
    favorites = []
    for row in cursor.fetchall():
        favorites.append({
            'id': row[0],
            'title': row[1],
            'type': row[2],
            'release_year': row[3],
            'director': row[4],
            'cast': row[5],
            'listed_in': row[6],
            'description': row[7],
            'query_used': row[8],
            'recommendation_date': row[9],
            'fav_id': row[10],
            'watched': bool(row[11]),
            'added_date': row[12]
        })
    
    conn.close()
    return favorites

def add_to_favorites(movie_id):
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    # Check if already in favorites
    cursor.execute('SELECT id FROM favorites WHERE movie_id = ?', (movie_id,))
    if not cursor.fetchone():
        cursor.execute('INSERT INTO favorites (movie_id) VALUES (?)', (movie_id,))
        conn.commit()
        success = True
    else:
        success = False
    
    conn.close()
    return success

def mark_as_watched(movie_id, rating=None, notes=None):
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    # Update favorites table
    cursor.execute('UPDATE favorites SET watched = TRUE WHERE movie_id = ?', (movie_id,))
    
    # Add to watch history
    cursor.execute('''
        INSERT INTO watch_history (movie_id, rating, notes)
        VALUES (?, ?, ?)
    ''', (movie_id, rating, notes))
    
    conn.commit()
    conn.close()

def get_recommendation_history():
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT query_used, COUNT(*) as count, MAX(recommendation_date) as last_used
        FROM recommended_movies
        GROUP BY query_used
        ORDER BY last_used DESC
        LIMIT 10
    ''')
    
    history = []
    for row in cursor.fetchall():
        history.append({
            'query': row[0],
            'count': row[1],
            'last_used': row[2]
        })
    
    conn.close()
    return history

# -------------------- Flask Routes -------------------- #

@app.route("/")
def cinemind_landing():
    return render_template("cinemind.html")

@app.route("/recommend", methods=["GET", "POST"])
def recommend_app():
    results = []
    query = ""
    movie_ids = []
    
    if request.method == "POST":
        query = request.form["query"]
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        
        # Convert pandas rows to dicts to avoid serialization issues
        results = [df.iloc[i].to_dict() for i in top_indices]
        
        # Save recommended movies to database and get their IDs
        movie_ids = save_recommended_movies(results, query)
    
    # Get recommendation history
    history = get_recommendation_history()
    
    return render_template("index.html", results=results, query=query, history=history, movie_ids=movie_ids)

@app.route("/favourites")
def favourites():
    favorites = get_favorites()
    return render_template("favourites.html", favorites=favorites)

@app.route("/plan")
def plan():
    favorites = get_favorites()
    return render_template("plan.html", favorites=favorites)

@app.route("/add_to_favorites", methods=["POST"])
def add_to_favorites_route():
    data = request.get_json()
    movie_id = data.get("movie_id")
    
    if add_to_favorites(movie_id):
        return jsonify({"success": True, "message": "Added to favorites!"})
    else:
        return jsonify({"success": False, "message": "Already in favorites!"})

@app.route("/mark_watched", methods=["POST"])
def mark_watched_route():
    data = request.get_json()
    movie_id = data.get("movie_id")
    rating = data.get("rating")
    notes = data.get("notes")
    
    mark_as_watched(movie_id, rating, notes)
    return jsonify({"success": True, "message": "Marked as watched!"})

# 🔥 Gemini AI-Powered Movie Planner Endpoint
@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    favorites = get_favorites()
    
    if not favorites:
        return jsonify({"plan": "No favorite movies found. Add some movies to your favorites first!"})

    # Count unwatched movies
    unwatched_movies = [m for m in favorites if not m['watched']]
    watched_movies = [m for m in favorites if m['watched']]
    
    if not unwatched_movies:
        return jsonify({"plan": "All your favorite movies have been watched! Add more movies to your favorites to get a new plan."})

    # Determine plan duration based on number of unwatched movies
    plan_days = min(len(unwatched_movies), 7)  # Max 7 days, or number of unwatched movies if less
    
    # Build prompt from movies
    unwatched_list = "\n".join([f"{i+1}. {m['title']} ({m['listed_in']})" for i, m in enumerate(unwatched_movies)])
    watched_list = "\n".join([f"{i+1}. {m['title']} ({m['listed_in']})" for i, m in enumerate(watched_movies)]) if watched_movies else "None"
    
    prompt = f"""
You are an intelligent movie scheduling assistant.

Here are the user's favorite movies:

UNWATCHED MOVIES ({len(unwatched_movies)} movies):
{unwatched_list}

WATCHED MOVIES ({len(watched_movies)} movies):
{watched_list}

Create a {plan_days}-day movie-watching plan (one movie per day). 
- Focus on the unwatched movies first
- If you have fewer than 7 unwatched movies, you can suggest re-watching some favorites
- For each day, select a movie, explain why it fits that day, and try to balance genres and moods
- Be creative, fun, and clear
- Consider the user's watching history and preferences
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        return jsonify({"plan": response.text})
    except Exception as e:
        return jsonify({"plan": f"Error: {str(e)}"})

# ------------------------------------------------------ #

if __name__ == "__main__":
    app.run(debug=True)
