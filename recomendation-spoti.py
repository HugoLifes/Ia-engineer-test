import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from collections import Counter
import random

#Leemos archivos csv en los dataframes

user_o_df = pd.read_csv('User_O.csv')
user_j_df = pd.read_csv('User_J.csv')
user_b_df = pd.read_csv('User_B.csv')
user_a_df = pd.read_csv('User_A.csv')
spotify_songs_df = pd.read_csv('spotify_songs.csv')

# Preprosesar datos

spotify_songs_df.dropna(subset=['track_name','track_artist'],inplace=True)
spotify_songs_df['track'] = spotify_songs_df['track_name'] + ' - ' + spotify_songs_df['track_artist']

for user_df in [user_o_df, user_j_df, user_b_df, user_a_df]:
    user_df['track'] = user_df['Song'] + ' - ' + user_df['Artist']
    

# Encontramos unas canciones comunes y preparamos datos para recomendaciones

user_dfs = [user_o_df, user_j_df, user_b_df, user_a_df]
user_common_dfs = []
user_tracks_list = []
user_features_list = []

for user_df in user_dfs:
    #if para saber si la cancion del suario se encuentra en la lista de canciones de spotify
    user_common_df = user_df[user_df['track'].isin(spotify_songs_df['track'])]
    user_common_dfs.append(user_common_dfs)
    user_tracks = user_common_df[['Id','track']]
    user_tracks_list.append(user_tracks)
    
    songs_features = spotify_songs_df[['track_id', 'track', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    user_features = pd.merge(songs_features,user_tracks, on='track')
    user_features.drop('track', axis=1, inplace=True)
    user_features_list.append(user_features)

# Escalar características y calcular similitud

scaler = StandardScaler()
songs_features_scaled = scaler.fit_transform(songs_features.select_dtypes(include='number'))

user_similarity_list = []
for user_features in user_features_list:
    user_features_scaled = scaler.transform(user_features.select_dtypes(include='number'))
    user_similarity = cosine_similarity(user_features_scaled, songs_features_scaled)
    user_similarity_list.append(user_similarity)

#Obtenemos recomendaciones basadas en similitud 

user_recommendations_list = []
for user_similarity in user_similarity_list:
    top_indices = user_similarity.argsort()[:, -5:][:, ::-1]
    recommendations = songs_features.iloc[top_indices.flatten()].drop_duplicates()
    user_recommendations_list.append(recommendations)

# Recomendamos playlists existentes

playlist_tracks = spotify_songs_df.groupby(['playlist_id', 'playlist_name'])['track_id'].apply(list).reset_index()

# Recomendar playlists existentes
playlist_tracks = spotify_songs_df.groupby(['playlist_id', 'playlist_name'])['track_id'].apply(list).reset_index()

def recommend_existing_playlist(user_tracks):
    playlist_tracks['common_tracks_count'] = playlist_tracks['track_id'].apply(lambda x: get_common_tracks_count(x, user_tracks['Id']))
    top_playlists = playlist_tracks.nlargest(2, 'common_tracks_count')[['playlist_name', 'common_tracks_count']]
    return top_playlists

def get_common_tracks_count(list1, list2):
    return len(set(list1).intersection(set(list2)))

user_existing_recommendations_list = []
for user_tracks in user_tracks_list:
    existing_recommendations = recommend_existing_playlist(user_tracks)
    user_existing_recommendations_list.append(existing_recommendations)

# Creamos las nuevas playlists
songs_features = songs_features.drop(['track_id', 'track'], axis=1)
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(songs_features_scaled)

def create_new_playlist(user_tracks, cluster_labels, all_song_data):
    user_track_ids = user_tracks['Id'].tolist()
    user_tracks_data = all_song_data[all_song_data['track_id'].isin(user_track_ids)]
    
    # Get indices of user tracks in all_song_data based on 'track_id'
    user_tracks_indices = all_song_data[all_song_data['track_id'].isin(user_tracks_data['track_id'])].index 
    
    user_cluster_labels = cluster_labels[user_tracks_indices]
    most_frequent_cluster = Counter(user_cluster_labels).most_common(1)[0][0]

    # Filter candidate tracks based on 'track_id'
    candidate_tracks = all_song_data[cluster_labels == most_frequent_cluster]['track_id']
    candidate_tracks = candidate_tracks[~candidate_tracks.isin(user_tracks_data['track_id'])]
    
    new_playlist_tracks = random.sample(list(candidate_tracks), 5)
    return all_song_data[all_song_data['track_id'].isin(new_playlist_tracks)]

user_new_playlist_list = []
for user_tracks in user_tracks_list:
    new_playlist = create_new_playlist(user_tracks, cluster_labels, spotify_songs_df)
    user_new_playlist_list.append(new_playlist)

# Se combinan las recomendaciones de usuario con el DataFrame spotify_songs_df para obtener información sobre el género y el subgénero.
for i in range(len(user_recommendations_list)):
    user_recommendations_list[i] = pd.merge(user_recommendations_list[i], spotify_songs_df[['track_id', 'playlist_genre', 'playlist_subgenre']], on='track_id', how='left')

#imprimimos las recomendaciones por usuario
user_names = ['User O', 'User J', 'User B', 'User A']
for i in range(len(user_names)):
    print(f"\nRecomendaciones para {user_names[i]}:")
    print("\nPlaylists Existentes:")
    print(user_existing_recommendations_list[i].to_markdown(index=False))
    print("\nNuevas Playlists:")
    print(user_new_playlist_list[i][['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre']].head(5).to_markdown(index=False))
    print("\nRecomendaciones basadas en similitud:")
    print(user_recommendations_list[i][['track', 'playlist_genre', 'playlist_subgenre']].head(5).to_markdown(index=False))