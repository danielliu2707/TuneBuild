import os
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from classes import FeatureEngineer, SpotipyPlaylist, Recommend
    
def main():
    # Firstly, get the user to authenticate
    CLIENT_ID = '9f951db530d8462fbedfd75507b90cbf'
    CLIENT_SECRET = '3f38813ebdb24d0caa8db79c5a169862'
    # VERY IMPORTANT: The scope dictates what you can retrieve from the user
    scope = "user-top-read playlist-modify-public playlist-modify-private"   
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, redirect_uri='http://localhost:5000/callback', client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
    user_id = sp.me()['id']
    
    # Secondly, get the user data
    # get_data = GetUserSongs(sp=sp)
    # get_data.get_identification()
    # get_data.add_track_features()
    # get_data.export_features('data/raw_user_data.csv')

    # Thirdly, test FeatureEngineer with all songs data
    all_songs = FeatureEngineer()
    all_songs.load_data('data/raw_allsongs_data.csv')
    all_songs.drop_duplicate_songs()
    all_songs.make_genres_list()
    all_songs.get_relevant_features()
    all_songs.export_songs('data/intermediate/song_df.csv')
    all_songs.get_tfidf()
    all_songs.normalize_features()
    all_songs_data = all_songs.get_final_df()
    
    # Fourthly, test FeatureEngineer with user songs data
    user_songs = FeatureEngineer()
    user_songs.load_data('data/raw_user_data.csv')
    user_songs.drop_duplicate_songs()
    user_songs.make_genres_list()
    user_songs.get_relevant_features()
    user_songs.get_tfidf()
    user_songs.normalize_features()
    user_songs_data = user_songs.get_final_df()
    
    # Fifth, test recommendation algorithm
    recommend = Recommend()
    recommend.load_all_songs('data/intermediate/song_df.csv')
    recommend.generate_playlist_feature(all_songs_data, user_songs_data)
    recommend.compute_cosine_similarity()
    
    # Sixth, get the recommended songs
    # NOTE: Used iteration argument so that if they want to create more and more playlists, it'll keep fetching the next 20 songs.
    # NOTE: Iteration starts at 0.
    recommend.get_top_20(iteration=0)
    recommended_songs = list(recommend.next_20_songs['id'])
    
    # Finally, create and add tracks to playlist
    playlist = SpotipyPlaylist(sp, user_id)
    playlist.create_playlist(playlist_name='TuneBuild Recommended Playlist', playlist_description='A TuneBuild playlist - Built by Daniel Liu.')
    playlist.add_to_playlist(recommended_songs)
    
if __name__ == "__main__":
    main()