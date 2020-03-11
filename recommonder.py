from dataset import Dataset
from matrix_factorization import MF
from memory_based import MemoryBase
import numpy as np
import pandas as pd
from numpy import linalg as la
import sys

# RATING_DATA = "./src/ratings.csv"
in_ = sys.argv[1]
operation = sys.argv[2]
based = sys.argv[3]
dataset = {'100k': './src/small/ratings.csv', '1m': './src/large/ratings.dat'}
RATING_DATA = dataset[in_]
#type_ = sys.argv[4]

training_data, testing_data, training_user, training_movie, testing_user, testing_movie = Dataset(RATING_DATA).separation()
# U,sigma,VT = la.svd(training_data)
# sigma = sigma**2    # 对奇异值求平方
# cnt = sum(sigma)    # 所有奇异值的和
# print(cnt)
# value = cnt*0.9     # 90%奇异值能量
# print(value)
# cnt2 = sum(sigma[:250])   # 2小于90%，前3个则大于90%，所以这里选择前三个奇异值
# print(cnt2)

if operation == "run":
    if based == 'model':
        model = MF(dataMatrix=training_data, num_features=1, learning_rate=0.001, beta=0.02, iterations=100, test=testing_data,
                   dict = [training_user, training_movie, testing_user, testing_movie])
        if type_ == 'test':
            print("+++++++++++++   Biased training    ++++++++++++++")
            model.train_biased()
            print(model.get_all_biased())
            model.biased_predict()
            model.plot_acc_biased()
            print("+++++++++++++   Unbiased training    ++++++++++++++")
            model.train_unbiased()
            print(model.get_all_unbiased())
            model.unbiased_predict()
            model.plot_acc_unbiased()
    elif based == 'memory':
        memory = MemoryBase(training_data, testing_data, training_movie, training_user, testing_movie, testing_user)
        if type_ == 'test':
            print("*************   User-Based    *****************")
            memory.user_tests()
            print("*************   Movie-Based    *****************")
            memory.movie_tests()

elif operation == "recommand":
    user = int(sys.argv[3])
    user_index = training_user.index(user)
    memory = MemoryBase(training_data, testing_data, training_movie, training_user, testing_movie, testing_user)
    top10 = memory.recommand(user_index)
    movies = pd.read_csv('./src/small/movies.csv')
    print("- " * 17, "Recommand  Results", " -" * 17, "-")
    for i in range(len(top10)):
        movie = top10[i]
        title = movies.loc[movies['movieId'] == movie, 'title'].iloc[0]
        print("|", "\t\t", i+1, "\t", title, " " * (50 - len(title)), "\t\t", "|")
    print("- " * 45)

