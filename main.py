import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="CineMind - Streamlit Demo",
    page_icon="🎬",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Loads the Netflix dataset and preprocesses the text data."""
    df = pd.read_csv("netflix_titles.csv")
    
    # Handle missing values gracefully
    df.fillna('', inplace=True)

    def create_text(row):
        return (f"{row['type']} {row['title']} {row['director']} {row['cast']} "
                f"{row['listed_in']} {row['description']}")
    
    df["textual_representation"] = df.apply(create_text, axis=1)
    return df

@st.cache_resource
def create_vectorizer_and_matrix(_df):
    """Creates and fits the TF-IDF vectorizer and computes the similarity matrix."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(_df["textual_representation"])
    return vectorizer, tfidf_matrix

# Load data and models
df = load_data()
vectorizer, tfidf_matrix = create_vectorizer_and_matrix(df)

# --- UI Components ---
st.title("🎬 CineMind - Streamlit Recommendation Demo")
st.markdown("""
Welcome to the lightweight Streamlit demonstration of the CineMind recommendation engine. 
This version focuses purely on the core machine learning model.

**For the full, feature-rich experience including AI planning and a futuristic UI, please run the main Flask application.**
""")
st.info("To run the main app: `python app.py`", icon="🚀")

st.header("Search for a Movie")
query = st.text_input(
    "Enter a movie theme, plot, genre, or title:", 
    "A space movie with a black hole"
)

if st.button("Get Recommendations", type="primary"):
    if query:
        # Transform the query using the fitted vectorizer
        query_vector = vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top 5 most similar items
        top_indices = similarities.argsort()[-5:][::-1]
        results = df.iloc[top_indices]

        st.header("🎯 Top Recommendations")

        for index, row in results.iterrows():
            with st.container():
                st.subheader(row['title'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(f"Type: {row['type']}")
                with col2:
                    st.text(f"Year: {row['release_year']}")
                with col3:
                    st.text(f"Rating: {row['rating']}")
                
                st.markdown(f"**Genres:** `{row['listed_in']}`")
                st.markdown(f"**Cast:** *{row['cast']}*")
                
                with st.expander("Read Description"):
                    st.write(row['description'])
                
                st.markdown("---")
    else:
        st.warning("Please enter a search query.")

st.sidebar.header("About CineMind")
st.sidebar.markdown("""
This project showcases a dual-implementation approach:

1.  **Flask App (`app.py`)**: A full-featured, visually rich application with a database, AI planner, and modern UI.
2.  **Streamlit App (`main.py`)**: This lightweight demo, focused on rapid prototyping and demonstrating the core ML model.

The Streamlit version is ideal for quickly testing the recommendation logic, while the Flask app provides a complete user experience.
""")
st.sidebar.header("Technology")
st.sidebar.markdown("""
- **Streamlit**: For the UI
- **Scikit-learn**: For TF-IDF and Cosine Similarity
- **Pandas**: For data manipulation
""")
