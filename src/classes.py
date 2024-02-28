# Import modules
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import spotipy
from spotipy.oauth2 import SpotifyOAuth

class FeatureEngineer():
    """
    Extracts all relevant features used in the recommendation algorithm from a dataset.
    """
    def __init__(self):
        """
        Initialize attributes.
        """
        self.song_df = None
        self.tfidf_df = None
    
    def load_data(self, path):
        """
        Loads input dataset

        Args:
            path (str): Path of file to be read in.
        """
        self.song_df = pd.read_csv(path)
        
    def drop_duplicate_songs(self):
        """
        Drops duplicate songs that exist due to different songs containing the same title.
        """
        self.song_df['artist_song'] = self.song_df.apply(lambda row: row['artist_name']+row['id'],axis = 1)
        self.song_df.drop_duplicates(subset='artist_song', inplace=True)
        self.song_df.reset_index(drop=True, inplace=True)
        
    def make_genres_list(self):
        """
        Obtain a list of genres by splitting the string of genres.
        """
        self.song_df['genres'] = self.song_df['genres'].apply(lambda row: row.split(' '))

    def get_relevant_features(self):
        """
        Extracting relevant features to be used for the recommendation algorithm.
        """
        self.song_df = self.song_df[['id', 'danceability', 'energy', 'key', 'loudness', 'mode',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'genres', 'artist_pop', 'track_pop']]
    
    def export_songs(self, path):
        """
        Exports output dataset.
        
        Args:
            path (str): Path of file to be exported
        """
        self.song_df.to_csv(path)
    
    def get_tfidf(self):
        """
        Converting genres into numeric values in order to allow for a useable format
        when inputting into the recommendation algorithm.
        """
        # Applying Tfidf on genres column
        tfidf = TfidfVectorizer()
        tfidf_array = tfidf.fit_transform(self.song_df['genres'].apply(lambda x: " ".join(x))).toarray()
        self.tfidf_df = pd.DataFrame(tfidf_array)
        self.tfidf_df.columns = ['genre' + '|' + i for i in tfidf.get_feature_names_out()]
        # Attempt to drop unknown genre if it exists
        try:
            self.tfidf_df.drop(columns='genre|unknown', inplace=True)
        except:
            pass
        self.tfidf_df.reset_index(drop=True, inplace=True)
    
    def normalize_features(self):
        """
        Normalizing features to ensure all features are on the same scale.
        """
        # Performing Normalization on popularities
        scaler = MinMaxScaler()
        self.song_df[['artist_pop', 'track_pop']] = scaler.fit_transform(self.song_df[['artist_pop', 'track_pop']])

        # Performing Normalization on floating columns
        float_col = self.song_df.loc[:, 'danceability':'tempo'].columns
        self.song_df[float_col] = scaler.fit_transform(self.song_df[float_col])
    
    # Concatenate features
    def get_final_df(self):
        """
        Return final dataset.
        """
        return pd.concat([self.song_df, self.tfidf_df], axis = 1)
    
class Recommend():
    """
    
    """
    def __init__(self):
        self.user_feature_sum = None
        self.all_feature_df = None
        self.all_songs_df = None
        self.songs_to_recommend = None
        self.next_20_songs = None
        
    def load_all_songs(self, path):
        self.all_songs_df = pd.read_csv(path)
    
    def generate_playlist_feature(self, all_song_features, user_songs_df):
        """
        Summarizes a users songs into a single vector

        Args:
            complete_feature_set (_type_): _description_
            user_songs_df (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get all features for user songs
        user_features_df = all_song_features[all_song_features['id'].isin(user_songs_df['id'].values)]
        user_features_df_final = user_features_df.drop(columns=['id', 'genres'])
        self.user_feature_sum = user_features_df_final.sum(axis=0)
        # Get all features for non user songs
        self.all_feature_df = all_song_features[~all_song_features['id'].isin(user_songs_df['id'].values)]
    
    def compute_cosine_similarity(self):
        # Keep only songs what aren't user songs to recommend
        self.songs_to_recommend = self.all_songs_df[self.all_songs_df['id'].isin(self.all_feature_df['id'].values)]
        # Obtain feature set of non user songs
        non_user_songs_features_arr = self.all_feature_df.drop(['id', 'genres'], axis = 1).values   # Convert it into an array
        # Find cosine similarity between non-user songs and complete song set
        cosine_scores = cosine_similarity(non_user_songs_features_arr, self.user_feature_sum.values.reshape(1, -1))[:, 0]
        self.songs_to_recommend['cosine_score'] = cosine_scores
    
    def get_top_20(self, iteration):
        """
        Get top 20 remaining recommended songs
        """
        upper_bound = (20 * iteration) + 20
        lower_bound = upper_bound-20
        self.next_20_songs = self.songs_to_recommend.sort_values('cosine_score', ascending=False).iloc[lower_bound:upper_bound]


class GetUserSongs():
    """
    Obtain a users' top 35 spotify songs and add relevant track features used
    for the recommendation algorithm.
    """
    def __init__(self, sp):
        """
        Initialize relevant attributes.
        """
        self.user_songs = None
        self.exported_df = None
        self.sp = sp
    
    def get_identification(self):
        """
        Get top 35 spotify songs and unique identifiers of the song.
        """
        artist_name = []
        track_id = []
        artist_uri = []
        for sp_range in ['short_term']:
            results = self.sp.current_user_top_tracks(time_range=sp_range, limit=35)
            for _, item in enumerate(results['items']):
                # Extract unique identifer of song
                artist_uri.append(item['artists'][0]['id'])
                # Extract additional relevant features
                artist_name.append(item['artists'][0]['name'])
                track_id.append(item['id'])
        
        # Construct dataframe of user top songs
        self.user_songs = pd.DataFrame({
        'artist_name': artist_name,
        'track_id': track_id,
        'artist_uri': artist_uri
        })
        
    def _extract_track_features(self, track_id: str):
        """
        Extracting features for each song to be used in the recommendation algorithm
        using the unique identifier for each song.

        Args:
            track_id (str): A unique identifier for the track
        """
        features = self.sp.audio_features(track_id)[0]

        # Add in track poularity:
        track_dict = self.sp.track(track_id)
        features['track_pop'] = track_dict['popularity']
        features['track_name'] = track_dict['name']

        # Add in artist genres
        artist = self.sp.track(track_id)['artists'][0]['id']
        artist_dict = self.sp.artist(artist)
        artist_genres = artist_dict['genres']
        features['artist_pop'] = artist_dict['popularity']
        if artist_genres:
            features['genres'] = " ".join([re.sub(' ','_',i) for i in artist_genres])
        else:
            features['genres'] = 'unknown'
        return features

    def add_track_features(self):
        """
        Adding extracted features from each song into a final dataframe.
        """
        feature_lst = []
        # Iterate through and get a list of features for each song
        with concurrent.futures.ThreadPoolExecutor() as exectuor:
            track_ids = list(self.user_songs['track_id'])
            feature_generator = exectuor.map(self._extract_track_features, track_ids)
            # Iterate through to get dictionary of all features for each song
            for feature in feature_generator:
                feature_lst.append(feature)
        # Making the features a df
        feature_df = pd.DataFrame(feature_lst)
        self.exported_df = pd.merge(self.user_songs, feature_df, left_on='track_id', right_on='id')
    
    def export_features(self, path):
        """
        Export final dataframe containing features of users' top songs.
        """
        self.exported_df.to_csv(path)
        
class SpotipyPlaylist:
    """
    Performs required actions on the users' spotify account.
    """
    def __init__(self, sp, user_id):
        """
        Initialize relevant attributes.
        """
        self.sp = sp
        self.playlist_id = None
        self.user_id = user_id
    
    def create_playlist(self, playlist_name: str, playlist_description: str):
        """
        Initializes an empty playlist on the users' Spotify account.

        Args:
            playlist_name (str): The title of the Spotify Playlist
            playlist_description (str): The description of the Spotify Playlist
        """
        self.sp.user_playlist_create(self.user_id, playlist_name, description = playlist_description)
        pl = list(self.sp.user_playlists(self.user_id)['items'])[0]
        self.playlist_id = pl['id']
        
    def add_to_playlist(self, playlist_tracks: list):
        """
        Adds recommendation trakcs on the users' Spotify account.

        Args:
            playlist_tracks (list): A list of track ID's to identify recommended songs
        """
        self.sp.playlist_add_items(playlist_id = self.playlist_id, items = playlist_tracks)

class Authorize():
    """
    Authenticates user to perform required actions on behalf of the user on their Spotify account.
    """
    def __init__(self, client_id, client_secret, scope, callback):
        """
        Initialize required attributes.
        """
        self.CLIENT_ID = client_id
        self.CLIENT_SECRET = client_secret
        self.scope = scope
        self.callback = callback
        self.user_id = None
        self.sp = None
        self.oauth = None
        
    def authorize(self):
        """
        Open authentication window and obtain access token to retrieve data from Spotify API.
        """
        self.oauth = SpotifyOAuth(scope=self.scope, redirect_uri=self.callback, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, open_browser=True, show_dialog=True)
        self.sp = spotipy.Spotify(auth_manager=self.oauth, auth = self.oauth.get_access_token()['access_token'])
        self.user_id = self.sp.me()['id']