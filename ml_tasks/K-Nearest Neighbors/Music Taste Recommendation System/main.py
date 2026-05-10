import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATASETS
# =========================
history = pd.read_csv("music_users.csv")
songs = pd.read_csv("songs.csv")

# =========================
# USER PROFILE BUILDING
# =========================

# Total listening behavior
user_profile = history.groupby("UserID").agg({
    "ListenCount": "sum",
    "MinutesPlayed": "sum"
}).reset_index()

# =========================
# FAVORITE GENRE PER USER
# =========================
merged = history.merge(songs, on="SongID")

favorite_genre = merged.groupby(["UserID", "Genre"]).size().reset_index(name="Count")
favorite_genre = favorite_genre.loc[favorite_genre.groupby("UserID")["Count"].idxmax()]
favorite_genre = favorite_genre[["UserID", "Genre"]]
favorite_genre.columns = ["UserID", "FavGenre"]

# =========================
# FAVORITE ARTIST PER USER
# =========================
favorite_artist = merged.groupby(["UserID", "Artist"]).size().reset_index(name="Count")
favorite_artist = favorite_artist.loc[favorite_artist.groupby("UserID")["Count"].idxmax()]
favorite_artist = favorite_artist[["UserID", "Artist"]]
favorite_artist.columns = ["UserID", "FavArtist"]

# =========================
# COMBINE PROFILE
# =========================
user_profile = user_profile.merge(favorite_genre, on="UserID")
user_profile = user_profile.merge(favorite_artist, on="UserID")

# =========================
# FEATURE ENGINEERING (KNN INPUT)
# =========================
X = user_profile[["UserID", "ListenCount", "MinutesPlayed"]]

# SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN KNN
# =========================
model = NearestNeighbors(n_neighbors=3, metric="cosine")
model.fit(X_scaled)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend(user_id):

    user_vec = X_scaled[user_profile["UserID"] == user_id]

    distances, indices = model.kneighbors(user_vec)

    similar_users = user_profile.iloc[indices[0]]["UserID"].values

    # songs already listened
    seen = set(history[history["UserID"] == user_id]["SongID"])

    # favorite genre of user
    fav_genre = user_profile[user_profile["UserID"] == user_id]["FavGenre"].values[0]

    # favorite artist of user
    fav_artist = user_profile[user_profile["UserID"] == user_id]["FavArtist"].values[0]

    # candidate songs from similar users
    candidate_songs = history[history["UserID"].isin(similar_users)]["SongID"].values

    print("\nPlaylist Recommendation for User", user_id)
    print("Favorite Genre:", fav_genre)
    print("Favorite Artist:", fav_artist)
    print("\nRecommended Songs:\n")

    printed = set()

    for song in candidate_songs:

        if song not in seen and song not in printed:

            song_info = songs[songs["SongID"] == song].iloc[0]

            # GENRE FILTER (important requirement)
            if song_info["Genre"] == fav_genre:

                print("-", song_info["SongName"], "by", song_info["Artist"])

                printed.add(song)

# =========================
# SEARCH ENGINE
# =========================
def search_song(keyword):

    result = songs[songs["SongName"].str.contains(keyword, case=False)]

    print("\nSearch Results:")
    print(result)

# =========================
# RUN SYSTEM
# =========================
while True:

    print("\n1. Recommend Songs")
    print("2. Search Song")
    print("3. Exit")

    choice = int(input("\nEnter choice: "))

    if choice == 1:
        uid = int(input("Enter User ID: "))
        recommend(uid)

    elif choice == 2:
        kw = input("Enter song name keyword: ")
        search_song(kw)

    else:
        break
# ======================================================

# The model learns:
# How to build user profiles based on listening history
# How to use KNN to find similar users and recommend songs
# How to implement a search function for songs
# How to handle user input and provide real-time recommendations


# Input Example:
# (1)
# (2)
# (3)
# (4)
# (5)
