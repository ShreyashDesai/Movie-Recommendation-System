# main.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Load dataset (MovieLens or custom CSV)
# Example CSV columns: movieId, title, genres
movies = pd.read_csv("movies.csv")

# Step 2: Content-Based Filtering (using genres)
# Convert genres into a bag-of-words
count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = count_vectorizer.fit_transform(movies['genres'])

# Compute similarity between movies
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to recommend movies based on a given movie
def recommend_content(movie_title, n=5):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movies.iloc[i[0]]['title'] for i in sim_scores[1:n+1]]
    return top_movies

# Step 3: Collaborative Filtering (user ratings)
# Example ratings.csv columns: userId, movieId, rating
ratings = pd.read_csv("ratings.csv")

# Create user-item matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Compute similarity between users
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))

# Function to recommend movies for a user
def recommend_collaborative(user_id, n=5):
    user_idx = user_movie_matrix.index.get_loc(user_id)
    sim_scores = list(enumerate(user_similarity[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Find most similar user
    similar_user_idx = sim_scores[1][0]
    similar_user_ratings = user_movie_matrix.iloc[similar_user_idx]
    
    # Recommend movies rated highly by similar user but not watched by current user
    user_ratings = user_movie_matrix.iloc[user_idx]
    recommendations = similar_user_ratings[user_ratings.isna()]
    top_movies = recommendations.sort_values(ascending=False).head(n).index
    return movies[movies['movieId'].isin(top_movies)]['title'].tolist()

# Example usage
print("Content-based recommendations for 'Toy Story (1995)':")
print(recommend_content("Toy Story (1995)"))

print("\nCollaborative recommendations for user 1:")
print(recommend_collaborative(1))
