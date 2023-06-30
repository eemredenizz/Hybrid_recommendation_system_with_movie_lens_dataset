
#############################################################################################
# HYBRID RECOMMENDER SYSTEM (Industrial Project)
#############################################################################################
######################################################################
# USER BASED RECOMMENDATION
######################################################################


#######################
#1- Veriyi Hazırlama
#######################
# Gerekli kütüohanelerin import edilmesi
import pandas as pd
pd.set_option("display.width", 500)
pd.set_option('display.expand_frame_repr', False)

# movie ve rating dataframe'lerinin birleştirilmesi
movie = pd.read_csv("movie_lens_dataset/movie.csv")
rating = pd.read_csv("movie_lens_dataset/rating.csv")

df = rating.merge(movie, how="left", on="movieId")

# 1000'den daha az yorum almış filmlerin kayda değer bir çıktısı olmayacağından dolayı çıkartıyoruz.
comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

# Pivot table'ın oluşturulması
user_movie_df = common_movies.pivot_table(index="userId", columns="title" ,values="rating")

# Sürecin fonksiyonlaştırılması
def preapering_data (movie_df, rating_df, min_comment=1000):
    # Gerekli kütüphanelerin import edilmesi
    import pandas as pd
    pd.set_option("display.width", 500)
    pd.set_option('display.expand_frame_repr', False)
    # !pip install surprise
    from surprise import Reader, SVD, Dataset, accuracy
    from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
    # rating ve movie dataframe'lerinin birleştirilmesi
    df = rating_df.merge(movie_df, how="left", on="movieId")
    # Çıktıyı bozacak filmlerin dataframe'den çıkarılması
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    #pivot table'ın oluşturlması
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")


######################
#2- Öneri yapılacak kullanıcın belirlenmesi
######################
random_user = 1320
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
#####################
#3- Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
#####################
movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
#####################
#4-Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#####################
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["user_id_1", "user_id_2"]

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})

recommendation_df[recommendation_df["weighted_rating"] > 3].sort_values(by="weighted_rating", ascending=False)

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3].sort_values(by="weighted_rating", ascending=False).head(5)

######################################################################
# ITEM BASED RECOMMENDATION
######################################################################

movie = pd.read_csv("movie_lens_dataset/movie.csv")
rating = pd.read_csv("movie_lens_dataset/rating.csv")
random_user = 1320

# Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'si
rating[(rating["userId"] == random_user) & (rating["rating"] == 5)].sort_values('timestamp',ascending=False).head(1)
movie_id = 4262
movie[movie["movieId"] == 4262]
movie_title = "Scarface (1983)"
movie_title = user_movie_df[movie_title]
user_movie_df.corrwith(movie_title).sort_values(ascending=False).head(6)

recommendation_list = ["Carlito's Way (1993)", "Casino (1995)", "Blow (2001)", "Presumed Innocent (1990)", "Boys from Brazil, The (1978)"]