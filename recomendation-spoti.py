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