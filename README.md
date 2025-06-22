# CineMind: Your Smart Movie Companion 🎬

CineMind is an intelligent movie recommendation and planning system designed to revolutionize your movie discovery experience. This project demonstrates a powerful, full-stack web application built with **Flask** for a rich, interactive user experience, and a complementary lightweight demonstration using **Streamlit**.

The Flask application is the core of this project, offering a visually appealing, futuristic interface with a comprehensive set of features, while the Streamlit app serves as a straightforward showcase of the underlying recommendation model.

[![CineMind Landing Page](https://user-images.githubusercontent.com/…..)](https://user-images.githubusercontent.com/…..)

---

## ✨ Key Features (Flask Application)

The primary Flask application is designed to be a full-fledged movie companion:

-   **🤖 AI-Powered Recommendations**: Utilizes `scikit-learn` with TF-IDF vectorization to provide semantic search capabilities, analyzing movie titles, genres, descriptions, and more.
-   **🧠 Intelligent AI Planner**: Integrates with **Google Gemini 2.0 Flash** to generate dynamic, personalized movie-watching schedules based on user preferences and viewing history.
-   **⭐ Favorites & Watch History**: Allows users to add movies to their favorites, mark them as watched, and add personal ratings and notes.
-   **💾 Persistent Database**: Uses **SQLite** to store all user data, including recommendations, favorites, and watch history, ensuring data persists across sessions.
-   ** futuristic UI/UX**: A sleek, modern interface built with Tailwind CSS, featuring glass morphism, neon glows, and smooth animations for an immersive user experience.
-   **📊 Search Analytics**: Tracks and displays recent search queries and recommendation statistics.

## 🚀 Implementations

This repository contains two distinct implementations:

1.  **Flask Application (Primary)**:
    -   **Visuals**: Highly stylized, interactive, and visually appealing.
    -   **Features**: Includes the full suite of features such as the AI planner, database storage, favorites management, and more.
    -   **File**: `app.py`
2.  **Streamlit Demonstration (Secondary)**:
    -   **Visuals**: A clean, simple, and functional interface.
    -   **Features**: A basic implementation focused on showcasing the core recommendation engine.
    -   **File**: `main.py`

The Flask application is the recommended way to experience CineMind, as it offers a superior user experience and a complete feature set.

---

## 👨‍💻 A Note from the Developer

This project was built with a specific philosophy in mind: start with a solid machine learning core and then use "vibe-driven" development to build an immersive and functional user experience around it.

### The "Model"

The recommendation engine isn't a traditionally trained deep learning model. Instead, it's an **unsupervised machine learning approach** that intelligently processes text data:

1.  **Data Corpus Creation**: All relevant text fields from the `netflix_titles.csv` (title, director, cast, description, genres) were concatenated into a single "document" for each movie.
2.  **Vectorization**: A **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** was fitted to this entire corpus. This "training" step involves scanning all the text to learn the vocabulary and calculate the IDF weights, which determine how important a word is in the context of the entire dataset.
3.  **Similarity Engine**: When a user enters a query, it's transformed into a TF-IDF vector using the already-fitted vectorizer. **Cosine Similarity** is then used to calculate the angle between the user's query vector and every movie vector in the dataset, allowing us to find the movies that are most contextually and semantically similar.

### "Vibe Coding" the Interface and Features

Once the core recommendation engine was functional, the rest of the application was built using what could be called "vibe coding." The goal was to create a **futuristic, intelligent, and seamless "CineMind" experience**.

-   **Aesthetic First**: Instead of building plain, functional pages, the development was driven by the desired aesthetic. A theme of neon glows, glass morphism, and fluid animations was established first.
-   **Feature Integration**: New features were added not just for functionality but to enhance the "smart companion" vibe.
    -   The **SQLite database** was added to give the app a memory and a sense of persistence.
    -   The **AI Planner** using Gemini 2.0 was the flagship feature to deliver on the promise of an "intelligent" planner, moving beyond simple recommendations.
-   **Iterative Refinement**: Each component, from buttons to movie cards, was iteratively styled and animated to fit this high-tech theme, ensuring the entire application felt cohesive and polished.

This approach prioritizes the user experience, allowing the "vibe" of the application to guide the implementation of its features and design.

---

## 🛠️ Technology Stack

-   **Backend**: Flask, Python
-   **Frontend**: HTML, Tailwind CSS, JavaScript
-   **Machine Learning**: Scikit-learn
-   **AI Integration**: Google Gemini 2.0 Flash
-   **Database**: SQLite
-   **Alternative UI**: Streamlit

## ⚙️ Setup and Installation

To run the Flask application locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/CineMind.git
    cd CineMind
    ```

2.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A `requirements.txt` file should be created for a production-ready app. For this project, key libraries are `flask`, `pandas`, `scikit-learn`, `google-generativeai`.*

3.  **Set up your Gemini API Key**:
    -   Open `app.py`.
    -   Replace `"your-gemini-api-key"` with your actual Gemini API key.

4.  **Run the Flask application**:
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

5.  **To run the Streamlit demo**:
    ```bash
    streamlit run main.py
    ```

## 📂 Project Structure

```
CineMind/
│
├── app.py                  # Main Flask application logic
├── main.py                 # Streamlit demonstration
├── netflix_titles.csv      # Dataset for movie recommendations
├── movies.db               # SQLite database file (created on run)
├── PROJECT_DESCRIPTION.txt # Detailed project overview
├── README.md               # This file
│
└── templates/
    ├── cinemind.html       # The main landing page
    ├── index.html          # The recommender app interface
    ├── favourites.html     # The user's favorites page
    └── plan.html           # The AI-generated movie plan page
```

---

This project showcases a blend of web development, machine learning, and cutting-edge AI to create a truly intelligent and user-friendly application. Feel free to explore, modify, and enhance it! 
