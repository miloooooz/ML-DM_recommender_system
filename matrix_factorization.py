import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

class MF:
    def __init__(self, dataMatrix, num_features, learning_rate, beta, iterations, test, dict):
        # Initialize the parameters
        self.data = dataMatrix
        self.num_users, self.num_movies = self.data.shape
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.beta = beta
        self.iterations = iterations
        self.test = test
        self.test_users, self.test_movies = self.test.shape
        self.train_user_dict = dict[0]
        self.train_movie_dict = dict[1]
        self.test_user_dict = dict[2]
        self.test_movie_dict = dict[3]

        # Initialize the weights
        self.u_matrix_biased = np.random.normal(scale=1. / self.num_features, size=(self.num_users, self.num_features))    # the matrix of users and features
        self.m_matrix_biased = np.random.normal(scale=1. / self.num_features, size=(self.num_movies, self.num_features))   # the matrix of features and movies
        self.u_matrix_unbiased = np.random.normal(scale=1. / self.num_features, size=(self.num_users, self.num_features))
        self.m_matrix_unbiased = np.random.normal(scale=1. / self.num_features, size=(self.num_movies, self.num_features))

        # Initialize the biases
        self.u_bias = np.zeros(self.num_users)
        self.m_bias = np.zeros(self.num_movies)
        self.bias = np.mean(self.data[np.where(self.data != 0)])

        self.samples = [
            (user_id, movie_id, self.data[user_id, movie_id])
            for user_id in range(self.num_users)
            for movie_id in range(self.num_movies)
            if self.data[user_id, movie_id] > 0
        ]

        # self.validate = [(user_index, movie_index, self.test[user_index, movie_index])
        #                  for user_index in range(self.test_users)
        #                  for movie_index in range(self.test_movies)
        #                  if self.test[user_index, movie_index] > 0]
        self.acc_biased = []
        self.acc_unbiased = []

    def train_biased(self):
        for i in range(self.iterations):
            self.cor = 0
            np.random.shuffle(self.samples)
            self.sgd_biased()
            mse = self.mse_biased()
            self.acc_biased.append(self.cor)

            # Print the result every 100 literations
            if (i + 1) % 10 == 0:
                print("Iteration: {}; Error: {:.6f}; Accuracy: {:.2f}".format(i + 1, mse, self.cor))
            if (i + 1) % 100 == 0:
                df = pd.DataFrame(data=self.get_all_biased().astype(float))
                df.to_csv('outfile_biased_mf.csv', float_format='%.6f', index=False)

    def train_unbiased(self):
        for i in range(self.iterations):
            self.cor_unbised = 0
            np.random.shuffle(self.samples)
            self.sgd_unbiased()
            mse = self.mse_unbiased()
            self.acc_unbiased.append(self.cor_unbised)

            # Print the result every 100 literations
            if (i + 1) % 10 == 0:
                print("Iteration: {}; Error: {:.6f}; Accuracy: {:.2f}".format(i + 1, mse, self.cor))
            if (i + 1) % 100 == 0:
                df = pd.DataFrame(data=self.get_all_unbiased().astype(float))
                df.to_csv('outfile_unbiased_mf.csv', float_format='%.6f', index=False)

    def sgd_biased(self):
        correct = 0
        for user_id, movie_id, rating in self.samples:
            # Calculate the squared error
            error = rating - self.get_rating_biased(user_id, movie_id)

            # Update weights
            self.u_matrix_biased[user_id, :] += self.learning_rate * (2 * error * self.m_matrix_biased[movie_id, :] - self.beta * self.u_matrix_biased[user_id, :])
            self.m_matrix_biased[movie_id, :] += self.learning_rate * (2 * error * self.u_matrix_biased[user_id, :] - self.beta * self.m_matrix_biased[movie_id, :])

            # Update bias
            self.u_bias[user_id] += self.learning_rate * (2 * error - self.beta * self.u_bias[user_id])
            self.m_bias[movie_id] += self.learning_rate * (2 * error - self.beta * self.m_bias[movie_id])

            if abs(error) <= 0.3:
                correct += 1
        self.cor = correct/len(self.samples) * 100

    def sgd_unbiased(self):
        correct = 0
        for user_id, movie_id, rating in self.samples:
            # Calculate the squared error
            error = rating - self.get_rating_unbiased(user_id, movie_id)

            # Update weights
            self.u_matrix_unbiased[user_id, :] += self.learning_rate * (2 * error * self.m_matrix_unbiased[movie_id, :] - self.beta * self.u_matrix_unbiased[user_id, :])
            self.m_matrix_unbiased[movie_id, :] += self.learning_rate * (2 * error * self.u_matrix_unbiased[user_id, :] - self.beta * self.m_matrix_unbiased[movie_id, :])

            if abs(error) <= 0.3:
                correct += 1
        self.cor_unbised = correct/len(self.samples) * 100

    # Calculate the mean square error for the entire matrix
    def mse_biased(self):
        mse = 0
        for user_id, movie_id, rating in self.samples:
            mse += (rating - self.get_rating_biased(user_id, movie_id)) ** 2
        return mse/len(self.samples)

    def mse_unbiased(self):
        mse = 0
        for user_id, movie_id, rating in self.samples:
            mse += (rating - self.get_rating_unbiased(user_id, movie_id)) ** 2
        return mse/len(self.samples)

    # Get the biased rating value of a particular user on a particular movie
    def get_rating_biased(self, user_id, movie_id):
        return self.bias + self.u_bias[user_id] + self.m_bias[movie_id] + self.u_matrix_biased[user_id, :].dot(self.m_matrix_biased[movie_id, :].T)

    # Get the unbiased rating value of a particular user on a particular movie
    def get_rating_unbiased(self, user_id, movie_id):
        return self.u_matrix_unbiased[user_id, :].dot(self.m_matrix_unbiased[movie_id, :].T)

    def get_user_matrix_biased(self):
        return self.u_matrix_biased

    def get_movie_matrix_biased(self):
        return self.m_matrix_biased

    def get_user_matrix_unbiased(self):
        return self.u_matrix_unbiased

    def get_movie_matrix_unbiased(self):
        return self.m_matrix_unbiased

    def get_bias(self):
        return self.bias, self.u_bias, self.m_bias

    # Get the estimate value of a particular user on a particular movie
    def get_result_biased(self, user_id, movie_id):
        result = self.get_all_biased()
        return result[user_id, movie_id]

    def get_result_unbiased(self, user_id, movie_id):
        result = self.get_all_unbiased()
        return result[user_id, movie_id]

    # Get the estimate value of the entire matrix
    def get_all_biased(self):
        return self.bias + self.u_bias[:, np.newaxis] + self.m_bias[np.newaxis:, ] + self.u_matrix_biased.dot(self.m_matrix_biased.T)

    def get_all_unbiased(self):
        return self.u_matrix_unbiased.dot(self.m_matrix_unbiased.T)

    def unbiased_predict(self):
        unbiased_cor = 0
        unbiased_mse = 0
        tot = 0
        for _, row in self.test.iterrows():
            user_index = int(row['userId'])
            movie_index = int(row['movieId'])
            rating = row['rating']
            try:
                user = self.train_user_dict.index(user_index)
                movie = self.train_movie_dict.index(movie_index)
                unbiased_estimate = self.get_result_unbiased(user, movie)
            except ValueError:
                unbiased_estimate = np.mean(self.data[user])
            tot += 1
            unbiased_err = abs(unbiased_estimate - rating)
            unbiased_mse += (rating - unbiased_estimate) ** 2
            if unbiased_err <= 0.3:
                unbiased_cor += 1
        unbiased_acc = unbiased_cor / tot * 100
        unbiased_rmse = sqrt(unbiased_mse / tot)
        print(f"==============The accuracy for unbiased MF is {unbiased_acc}.==============")
        print(f"==============The RMSE for unbiased MF is {unbiased_rmse}.==============")

    def biased_predict(self):
        biased_cor = 0
        biased_mse = 0
        tot = 0
        for _, row in self.test.iterrows():
            user_index = int(row['userId'])
            movie_index = int(row['movieId'])
            rating = row['rating']
            try:
                user = self.train_user_dict.index(user_index)
                movie = self.train_movie_dict.index(movie_index)
                biased_estimate = self.get_result_biased(user, movie)
            except ValueError:
                biased_estimate = np.mean(self.data[user])
            tot += 1
            biased_err = abs(biased_estimate - rating)
            biased_mse += (rating - biased_estimate) ** 2
            if biased_err <= 0.3:
                biased_cor += 1
        # for user_index in range(self.test.shape[0]):
        #     for movie_index in range(self.test.shape[1]):
        #         try:
        #             rating = self.test.iloc(user_index, movie_index)
        #             if rating != 0:
        #                 user = self.train_user_dict.index(self.test_user_dict[user_index])
        #                 movie = self.train_movie_dict.index(self.test_movie_dict[movie_index])
        #                 biased_estimate = self.get_result_biased(user, movie)
        #                 tot += 1
        #             else:
        #                 biased_estimate = 0
        #         except ValueError:
        #             biased_estimate = 0
        #             rating = 0
        #         biased_err = abs(biased_estimate - rating)
        #         biased_mse += (rating-biased_estimate) ** 2
        #         if biased_err <= 0.3:
        #             biased_cor += 1
        biased_acc = biased_cor/tot * 100
        biased_rmse = sqrt(biased_mse/tot)
        print(f"==============The accuracy for biased MF is {biased_acc}.==============")
        print(f"==============The RMSE for biased MF is {biased_rmse}.==============")

    # Plot the figure of accuracy
    def plot_acc_biased(self):
        plt.plot(np.arange(self.iterations), self.acc_biased)
        plt.title(f"Biased MF with Learning Rate: {self.learning_rate}; Feature: {self.num_features}")
        plt.ylabel("%Accuracy")
        plt.xlabel("Iterations")
        plt.show()

    def plot_acc_unbiased(self):
        plt.plot(np.arange(self.iterations), self.acc_unbiased)
        plt.title(f"Unbiased MF with Learning Rate: {self.learning_rate}; Feature: {self.num_features}")
        plt.ylabel("%Accuracy")
        plt.xlabel("Iterations")
        plt.show()
