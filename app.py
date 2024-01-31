from flask import Flask, render_template, url_for, redirect, jsonify, session, request
from datetime import datetime
import requests
import urllib.parse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
from src.classes import GetUserSongs, Authorize, FeatureEngineer, Recommend, SpotipyPlaylist

app = Flask(__name__, static_folder="templates/assets")

# Needed to access Flask Session (can store data accessed later between requests).
app.secret_key = os.urandom(64)

REDIRECT_URI = 'http://localhost:5000/callback'

# URL's to get the token from spotify, refresh token and API's base URL
TOKEN_URL = SpotifyOAuth.OAUTH_TOKEN_URL
API_BASE_URL = 'https://api.spotify.com/v1/'

# Set cache handler for storing access tokens
cache_handler = spotipy.cache_handler.FlaskSessionCacheHandler(session)

# Authorize
scope = 'user-top-read playlist-modify-public playlist-modify-private'

# Load all env variables (client_id, client_secret)
load_dotenv()

authorize = Authorize(client_id=os.getenv('CLIENT_ID'),client_secret=os.getenv('CLIENT_SECRET'),
                      scope="user-top-read playlist-modify-public playlist-modify-private", callback='http://localhost:5000/callback')
authorize.authorize()

# Setting constant variables
oauth_manager = authorize.oauth
sp = authorize.sp
user_id = authorize.user_id

@app.route("/")
@app.route("/home")
def home():
    # If no access token is not session, force user to authenticate
    if 'access_token' not in session:
        return render_template('home.html')
    
    # If access token has expired, refresh the token
    if datetime.now().timestamp() > session['expires_at']:
        return redirect(url_for('refresh_token'))
    return redirect(url_for('get_songs'))

@app.route("/spotify-auth")
def spotify_auth():
    """
     Makes request to Spotifies Auth URL, passing params to retrieve playlists and 
     redirect the user to this authentication URL.
    """
    return redirect(oauth_manager.get_authorize_url())

""" 
 When logging in, either the user will login successfully - Spotify gives us a code to get an access token.
 The user may login unsuccessfully - We will get an error.
 
 callback endpoint: In an OAuth process, used to redirect the user back to the client application
 once they've been granted permission.
 * Once we get user info, Spotify will callback to this /callback endpoint.
"""

@app.route('/callback')
def callback():
    # If we get a code, get access token
    if request.args.get('code'):
        tokens = oauth_manager.get_access_token(request.args.get('code'))
        session['access_token'] = tokens['access_token']
        session['refresh_token'] = tokens['refresh_token']
        session['expires_at'] = tokens['expires_at']
        return redirect(url_for('get_songs'))
    # Otherwise, reauthenticate
    else:
        return redirect(url_for('spotify-auth'))
        

@app.route('/refresh-token')
def refresh_token():
    # If no refresh token, request a login
    if 'refresh_token' not in session:
        return redirect(url_for('spotify_auth'))
    
    # If access token has expired, make a request for a fresh access token
    if datetime.now().timestamp() > session['expires_at']:
        new_token_info = oauth_manager.refresh_access_token(session['refresh_token'])
        session['access_token'] = new_token_info['access_token']
        session['expires_at'] = new_token_info['expires_at']
        
        return redirect(url_for('get_songs'))

@app.route("/get_songs")
def get_songs():
    """
    Renders create-playlist page
    """
    
    # Get user data
    get_data = GetUserSongs(sp=sp)
    get_data.get_identification()
    get_data.add_track_features()
    get_data.export_features('data/raw_user_data.csv')
    
    return render_template('create-playlist.html')

@app.route("/create_playlist")
def create_playlist():
    if 'access_token' not in session:
        return redirect(url_for('spotify_auth'))
    
    # If access token has expired, refresh the token
    if datetime.now().timestamp() > session['expires_at']:
        print('Token Expired')
        return redirect(url_for('refresh_token'))
    
    # FeatureEngineer with all songs data
    all_songs = FeatureEngineer()
    all_songs.load_data('data/raw_allsongs_data.csv')
    all_songs.drop_duplicate_songs()
    all_songs.make_genres_list()
    all_songs.get_relevant_features()
    all_songs.export_songs('data/intermediate/song_df.csv')
    all_songs.get_tfidf()
    all_songs.normalize_features()
    all_songs_data = all_songs.get_final_df()
    
    # FeatureEngineer with user songs data
    user_songs = FeatureEngineer()
    user_songs.load_data('data/raw_user_data.csv')
    user_songs.drop_duplicate_songs()
    user_songs.make_genres_list()
    user_songs.get_relevant_features()
    user_songs.get_tfidf()
    user_songs.normalize_features()
    user_songs_data = user_songs.get_final_df()
    
    # Run recommendation algorithm
    recommend = Recommend()
    recommend.load_all_songs('data/intermediate/song_df.csv')
    recommend.generate_playlist_feature(all_songs_data, user_songs_data)
    recommend.compute_cosine_similarity()
    
    # Get top 20 recommended songs
    # NOTE: Used iteration argument so that if they want to create more and more playlists, it'll keep fetching the next 20 songs.
    # NOTE: Iteration starts at 0.
    recommend.get_top_20(iteration=0)
    recommended_songs = list(recommend.next_20_songs['id'])
    
    # Create and add tracks to playlist on user account
    playlist = SpotipyPlaylist(sp, user_id)
    playlist.create_playlist(playlist_name='TuneBuild Recommended Playlist', playlist_description='A TuneBuild playlist - Built by Daniel Liu.')
    playlist.add_to_playlist(recommended_songs)

    return render_template('create-playlist.html')

if __name__ == '__main__':
    app.run(debug=True)
