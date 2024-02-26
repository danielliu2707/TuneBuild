from flask import Flask, render_template, url_for, redirect, session, request, jsonify
from datetime import datetime
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
from src.classes import GetUserSongs, Authorize, FeatureEngineer, Recommend, SpotipyPlaylist
import jsonpickle


app = Flask(__name__, static_folder="templates/assets")

app.secret_key = os.urandom(64)

REDIRECT_URI = 'http://localhost:5000/callback'
TOKEN_URL = SpotifyOAuth.OAUTH_TOKEN_URL
API_BASE_URL = 'https://api.spotify.com/v1/'

# Load all env variables (client_id, client_secret)
load_dotenv()

# Setting constant variables
# oauth_manager = authorize.oauth
# sp = authorize.sp
# user_id = authorize.user_id

# authorize = Authorize(client_id=session.get('client_id'),client_secret=session.get('client_secret'),
#                     scope="user-top-read playlist-modify-public playlist-modify-private", callback='http://localhost:5000/callback')
# authorize.authorize()

@app.route("/")
@app.route("/home")
def home():
    """
    Endpoint that loads the home page.
    """
    # If no access token in session, force user to authenticate
    if 'access_token' not in session:
        return render_template('home.html')
    
    # If access token has expired, refresh the token
    if datetime.now().timestamp() > session['expires_at']:
        return redirect(url_for('refresh_token'))
    return redirect(url_for('get_songs'))

@app.route('/collect-data', methods=['POST'])
def my_form_post():
    client_details = request.json
    # Authorize user
    authorize = Authorize(client_id=client_details['client-id'],client_secret=client_details['client-secret'],
                        scope="user-top-read playlist-modify-public playlist-modify-private", callback='http://localhost:5000/callback')
    authorize.authorize()
    
    session['oauth_manager'] = jsonpickle.encode(authorize.oauth)
    session['sp'] = jsonpickle.encode(authorize.sp)
    session['user_id'] = authorize.user_id
    
    # I WANT TO GET RID OF THIS!!! TRY NOT TO USE GLOBAL VARIABLES
    # TODO: Since sessions don't work, try a database.
    # Or figure out why sessions don't translate over
    global oauth_manager
    oauth_manager = jsonpickle.encode(authorize.oauth)
    global sp 
    sp = jsonpickle.encode(authorize.sp)
    global user_id
    user_id = authorize.user_id
    
    
    return redirect(url_for('spotify_auth'))

@app.route("/spotify-auth")
def spotify_auth():
    """
    Endpoint used in redirecting user to spotify authentication
    """
    return jsonify({'url': jsonpickle.decode(session.get('oauth_manager')).get_authorize_url()})

""" 
 When logging in, either the user will login successfully - Spotify gives us a code to get an access token.
 The user may login unsuccessfully - We will get an error.
 
 callback endpoint: In an OAuth process, used to redirect the user back to the client application
 once they've been granted permission.
 * Once we get user info, Spotify will callback to this /callback endpoint.
"""

@app.route('/callback')
def callback():
    """
    Endpoint used to redirect user back to the client application once authenticated.
    Then obtains the access_token used for making calls to the Spotify API, 
    refresh_token and expiry of the access_token.
    """
    # I WANT TO GET RID OF THIS!!! BEST NOT TO USE GLOBAL VARIABLES
    session['oauth_manager'] = oauth_manager
    session['sp'] = sp
    session['user_id'] = user_id
    print(session)
    
    
    # If we get a code, get access token
    if request.args.get('code'):
        tokens = jsonpickle.decode(session.get('oauth_manager')).get_access_token(request.args.get('code'))
        session['access_token'] = tokens['access_token']
        session['refresh_token'] = tokens['refresh_token']
        session['expires_at'] = tokens['expires_at']
        return redirect(url_for('get_songs'))
    # Otherwise, reauthenticate
    else:
        return redirect(url_for('spotify_auth'))
        

@app.route('/refresh-token')
def refresh_token():
    # If no refresh token, request a login
    if 'refresh_token' not in session:
        return redirect(url_for('spotify_auth'))
    
    # If access token has expired, make a request for a fresh access token
    if datetime.now().timestamp() > session['expires_at']:
        new_token_info = jsonpickle.decode(session.get('oauth_manager')).refresh_access_token(session['refresh_token'])
        session['access_token'] = new_token_info['access_token']
        session['expires_at'] = new_token_info['expires_at']
        
        return redirect(url_for('get_songs'))

@app.route("/get_songs")
def get_songs():
    """
    Endpoint used for obtaining features of the users' top songs in 
    addition to rendering the create-playlist page.
    """
    # Get user data
    get_data = GetUserSongs(sp=jsonpickle.decode(session.get('sp')))
    get_data.get_identification()
    get_data.add_track_features()
    get_data.export_features('data/raw_user_data.csv')
    
    return render_template('create-playlist.html')

@app.route("/create_playlist")
def create_playlist():
    """
    Endpoint used for creating the recommended playlist on the users Spotify
    when a user clicks the CREATE PLAYLIST button.
    """
    # If access token not found, reauthenticate
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
    playlist = SpotipyPlaylist(jsonpickle.decode(session.get('sp')), session.get('user_id'))
    playlist.create_playlist(playlist_name='TuneBuild Recommended Playlist', playlist_description='A TuneBuild playlist - Built by Daniel Liu.')
    playlist.add_to_playlist(recommended_songs)

    return render_template('create-playlist.html')

if __name__ == '__main__':
    app.run(debug=True)
