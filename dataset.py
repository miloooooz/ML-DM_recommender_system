import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

rating_col_to_drop = ['timestamp']
# movie_data = './src/movies.csv'
# movie_col_to_drop = ['title']
# tag_data = './src/tags.csv'
# tag_col_to_drop = ['userId', 'timestamp']


class Dataset:
    def __init__(self, rating_data):
        if rating_data == './src/large/ratings.dat':
            self.rating_ds = pd.read_csv(rating_data, sep='::', encoding='latin-1', header=None, engine='python')
            self.rating_ds.columns = ['userId', 'movieId', 'rating', 'timestamp']
            self.rating_df = pd.DataFrame(self.rating_ds.drop(rating_col_to_drop, axis=1))
        else:
            self.rating_ds = pd.read_csv(rating_data)
            self.rating_df = pd.DataFrame(self.rating_ds.drop(rating_col_to_drop, axis=1), columns=['userId', 'movieId', 'rating'])
        # self.movie_ds = pd.read_csv(movie_data)
        # self.movie_df = pd.DataFrame(self.movie_ds.drop(movie_col_to_drop, axis=1), columns=['movieId', 'genres'])
        # self.tag_ds = pd.read_csv(tag_data)
        # self.tag_df = pd.DataFrame(self.tag_ds.drop(tag_col_to_drop, axis=1), columns=['movieId', 'tag'])
        # self.movie_df['movieId'] = pd.to_numeric(self.movie_df['movieId'])

    def separation(self):
        # separate training and testing data wp 0.8, 0.2
        self.training_data, self.testing_data = train_test_split(self.rating_df, test_size=0.2, random_state=43)

        self.training_user_dict = list(set(sorted([i for i in self.training_data.userId])))
        self.training_movie_dict = list(set(sorted([i for i in self.training_data.movieId])))
        self.testing_user_dict = list(set(sorted([i for i in self.testing_data.userId])))
        self.testing_movie_dict = list(set(sorted([i for i in self.testing_data.movieId])))

        self.training_matrix = np.zeros(shape=(len(set(self.training_data.userId)), len(set(self.training_data.movieId))))
        for _, row in self.training_data.iterrows():
            user = int(row['userId'])
            movie = int(row['movieId'])
            rating = row['rating']
            user_index = self.training_user_dict.index(user)
            movie_index = self.training_movie_dict.index(movie)
            self.training_matrix[user_index,movie_index] = rating

        # self.testing_matrix = np.zeros(shape=(len(set(self.testing_data.userId)), len(set(self.testing_data.movieId))))
        # for _, row in self.testing_data.iterrows():
        #     user = int(row['userId'])
        #     movie = int(row['movieId'])
        #     rating = row['rating']
        #     user_index = self.testing_user_dict.index(user)
        #     movie_index = self.testing_movie_dict.index(movie)
        #     self.testing_matrix[user_index, movie_index] = rating

        return self.training_matrix, self.testing_data, self.training_user_dict, self.training_movie_dict, self.testing_user_dict, self.testing_movie_dict



