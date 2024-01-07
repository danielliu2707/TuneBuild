# Extract the relevant audio features from each song
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import re

CLIENT_ID = '9f951db530d8462fbedfd75507b90cbf'
CLIENT_SECRET = '3f38813ebdb24d0caa8db79c5a169862'
scope = "user-library-read"

def extract_track_features(track_uri: str):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, redirect_uri='http://localhost:5000/callback', client_id=CLIENT_ID, client_secret=CLIENT_SECRET))
    
    # Audio Features     ## NOTE: Could consider other numeric calculations - Go through Spotipy documentation
    features = sp.audio_features(track_uri)[0]

    # Add in track poularity:
    features['track_pop'] = sp.track(track_uri)['popularity']

    # Add in artist genres
    artist = sp.track(track_uri)['artists'][0]['id']
    artist_genres = sp.artist(artist)['genres']
    if artist_genres:
        features['artist_genres'] = " ".join([re.sub(' ','_',i) for i in artist_genres])
    else:
        features['artist_genres'] = 'unavailable'

    return features