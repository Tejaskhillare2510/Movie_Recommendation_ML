
import streamlit as st
import pickle

#Using KNN Model

modelknn_file_path = 'model_knn.pkl'
with open(modelknn_file_path, 'rb') as f:
    model_knn = pickle.load(f)
    
moviefeatures_file_path = 'movie_features_df.pkl'
with open(moviefeatures_file_path, 'rb') as f:
    movie_features_df = pickle.load(f)    


def getMovieIndex( movie_title):
    for i in movie_features_df:
        if movie_features_df.index[i] == movie_title:
            return i
    return None

def Recommend_movies_knn( movie_title):
    # Get the index of the specified movie title
    query_index = getMovieIndex(movie_title)
    
    # Use the KNN model to find nearest neighbors
    distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
    
    recommended_movies = []
    for i in range(len(distances.flatten())):
        if i == 0:
            print(f"Recommendations for {movie_features_df.index[query_index]}:\n")
        else:
            recommended_movie = movie_features_df.index[indices.flatten()[i]]
            recommended_distance = distances.flatten()[i]
            recommended_movies.append((recommended_movie, recommended_distance))
    
    return recommended_movies


# Using SVD Model

modelsvd_file_path = 'model_svd.pkl'
with open(modelsvd_file_path, 'rb') as f:
    model_svd = pickle.load(f)
    
moviesdata_file_path = 'movies_data.pkl'
with open(moviesdata_file_path, 'rb') as f:
    movies_data = pickle.load(f)

def Recommend_movies_svd(movie_title, top_n=6):
   
    movie_id = movies_data.loc[movies_data['title'] == movie_title, 'movieId'].iloc[0]
    

    movie_factors = model_svd.qi[model_svd.trainset.to_inner_iid(movie_id)]
    
    similarities = []
    for inner_id in model_svd.trainset.all_items():
        other_movie_factors = model_svd.qi[inner_id]
        similarity = sum(movie_factors * other_movie_factors)
        similarities.append((model_svd.trainset.to_raw_iid(inner_id), similarity))
    
    # Sort movies by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommended movie IDs (excluding the input movie)
    recommended_movie_ids = [sim[0] for sim in similarities[:top_n] if sim[0] != movie_id]
    
    # Get movie titles based on recommended IDs
    recommended_movies = movies_data[movies_data['movieId'].isin(recommended_movie_ids)]['title'].values
    
    return recommended_movies



# Streamlit App
st.title('Movie Recommendation System')

# User Input for Movie Title
user_input = st.text_input('Enter a movie title:', '')

if st.button('Get Recommendations'):
    # Recommendations from KNN Model
    st.header('Recommendations using KNN Model')
    knn_recommendations = Recommend_movies_knn(user_input)
    if knn_recommendations:
        st.write('KNN Recommendations:')
        for i, (movie, distance) in enumerate(knn_recommendations, 1):
            st.write(f'{i}. {movie} (Distance: {distance:.2f})')
    else:
        st.write('No KNN Recommendations found.')

    # Recommendations from SVD Model
    st.header('Recommendations using SVD Model')
    svd_recommendations = Recommend_movies_svd(user_input)
    if len(svd_recommendations) > 0:
        st.write('SVD Recommendations:')
        for i, movie in enumerate(svd_recommendations, 1):
            st.write(f'{i}. {movie}')
    else:
        st.write('No SVD Recommendations found.')