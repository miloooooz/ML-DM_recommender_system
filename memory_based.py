timport pandas as pd
import numpy as np
from numpy import linalg as la
from math import sqrt
import operator
from sklearn.metrics.pairwise import pairwise_distances


class MemoryBase:
    def __init__(self, train, test, train_movie_dict, train_user_dict, test_movie_dict, test_user_dict):
        self.train = train
        self.test = test
        self.train_movie_dict = train_movie_dict
        self.train_user_dict = train_user_dict
        self.test_movie_dict = test_movie_dict
        self.test_user_dict = test_user_dict
        self.sample_matrix = [
            (user_id, movie_id, self.train[user_id, movie_id])
            for user_id in range(len(self.train_user_dict))
            for movie_id in range(len(self.train_movie_dict))
            if self.train[user_id, movie_id] > 0
        ]

    def svd(self):
        U,Sigma,VT = la.svd(self.train)

        # 重构矩阵
        dig = np.mat(np.eye(250)*Sigma[:250]) # 获得对角矩阵
        # dim = data.T * U[:,:count] * dig.I # 降维 格外变量这里没有用
        redata = U[:,:250] * dig * VT[:250,:]   # 重构
        print(self.train.shape)
        self.train = redata
        print(self.train.shape)

    def cal_similarity_user(self):
        # Here we use cosine similarity
        self.user_cor = 1 - pairwise_distances(self.train.astype('float16'), metric='cosine')
        return self.user_cor

    def cal_similarity_movie(self):
        # Here we use cosine similarity
        self.movie_cor = 1 - pairwise_distances(self.train.T.astype('float16'), metric='cosine')
        return self.movie_cor

    def user_predict(self, user_index, movie, user_sim, similarity):
        sim_user_nb = 10
        top_users = {}
        mean_rating = np.mean(self.train[user_index])
        try:
            movie_index = self.train_movie_dict.index(movie)        # since 'movie' is the movieId and we need to transform it from the real Id to the index in trainning set
            i = 0       # current number of user who watched the movie
            ind = 0     # current index of user
            estimate = 0        # estimate of ratings

            # Find the top 10 similar user and store the similarity
            while i < sim_user_nb:
                if ind == len(self.train_user_dict):
                    break
                if (self.train[user_sim[ind]][movie_index] > 0) and (similarity[user_index][user_sim[ind]] > 0):
                    top_users[user_sim[ind]] = similarity[user_index][user_sim[ind]]
                    i += 1
                    ind += 1
                else:
                    ind += 1
            if len(top_users) == 0:
                estimate_rating = mean_rating
            else:
                sum_factor = 0
                for user_i in top_users:
                    sum_factor += top_users[user_i]
                    estimate += self.train[user_i][movie_index] * top_users[user_i]
                estimate_rating = estimate / sum_factor
        except ValueError:
            estimate_rating = mean_rating
        return estimate_rating

    def user_predict2(self, user_index, movie, user_sim, similarity):
        top_users = {}
        mean_rating = np.mean(self.train[user_index])
        try:
            movie_index = self.train_movie_dict.index(movie)        # since 'movie' is the movieId and we need to transform it from the real Id to the index in trainning set
            for user_i in range(len(self.train_user_dict)):
                if user_i != user_index:
                    a = 0
                    b = 0
                    c = 0
                    mean2 = np.mean(self.train[user_i])
                    for movie_i in range(len(self.train_movie_dict)):
                        if self.train[user_i][movie_i] != 0 and self.train[user_index][movie_i] != 0:
                            a += (self.train[user_index][movie_i] - mean_rating) * (self.train[user_i][movie_i] - mean2)
                            b += (self.train[user_index][movie_i] - mean_rating) ** 2
                            c += (self.train[user_i][movie_i] - mean2) ** 2
                    if b != 0 and c != 0:
                        top_users[user_i] = a / (sqrt(b) * sqrt(c))
            if len(top_users) == 0:
                estimate_rating = mean_rating
            else:
                sum_factor = 0
                estimate = 0
                for user_i in top_users:
                    sum_factor += abs(top_users[user_i])
                    estimate += (self.train[user_i][movie_index] - np.mean(self.train[user_i])) * top_users[user_i]
                estimate_rating = mean_rating + estimate / sum_factor

        except ValueError:
            estimate_rating = mean_rating
        return estimate_rating

    def sim(self):
        def pearsSim(inA,inB):
            if(len(inA) < 3):
                return 1.0
            return 0.5 + 0.5 * np.corrcoef(inA,inB,rowvar=0)[0][1] 
        dataMat = self.train.T
        n = np.shape(dataMat)[1]
        U,Sigma,VT = la.svd(dataMat)
        Sig = np.mat(np.eye(250)*Sigma[:250])  #将奇异值向量转换为奇异值矩阵
        #xformedItems = self.train.T * U[:,:250] * Sig.I  # 降维方法 通过U矩阵将物品转换到低维空间中 （商品数行x选用奇异值列）
        xformedItems = np.matmul(np.matmul(dataMat.T, U[:,:250]), Sig.I)

        sim = np.zeros((len(self.train_user_dict), len(self.train_user_dict)))
        for i in range(len(self.train_user_dict)):
            for j in range (len(self.train_user_dict)):
                sim[i,j] = pearsSim(xformedItems[i,:].T, xformedItems[j,:].T)

        return sim

    def pearson(self):
        def pearsSim(inA,inB):
            if(len(inA) < 3):
                return 1.0
            return 0.5 + 0.5 * np.corrcoef(inA,inB,rowvar=0)[0][1] 

        sim = np.zeros((len(self.train_user_dict), len(self.train_user_dict)))
        for i in range(len(self.train_user_dict)):
            for j in range (len(self.train_user_dict)):
                sim[i,j] = pearsSim(self.train[i,:].T, self.train[j,:].T)

        return sim

    def user_predict3(self, user_index, movie, user_sim, similarity):
        mean_rating = np.mean(self.train[user_index])
        simTotal = 0.0
        ratSimTotal = 0.0
        try:
            movie_index = self.train_movie_dict.index(movie)
            for i in range(len(self.train_user_dict)):
                userRating = self.train[i, movie_index]
                if userRating == 0 or i == user_index:
                    continue
                # 这里需要说明：由于降维后的矩阵与原矩阵代表数据不同（行由用户变为了商品），所以在比较两件商品时应当取【该行所有列】 再转置为列向量传参
                sim = similarity[i, user_index]
                # print('%d 和 %d 的相似度是: %f' % (item, j, similarity))
                simTotal += sim
                ratSimTotal += sim * userRating
            if simTotal == 0:
                return 0
            else:
                return ratSimTotal/simTotal
        except ValueError:
            return mean_rating



    def movie_predict(self, user_index, movie_index, similarity, movie_sim):
        sim_movie_nb = 15
        top_movies = {}
        mean_rating = np.mean(self.train[user_index])
        try:
            i = 0
            ind = 0
            estimate = 0
            while i < sim_movie_nb:
                if ind == len(self.train_movie_dict):
                    break
                if (self.train[user_index][movie_sim[ind]] > 0) and (similarity[movie_index][movie_sim[ind]] > 0):
                    top_movies[movie_sim[ind]] = similarity[movie_index][movie_sim[ind]]
                    i += 1
                    ind += 1
                else:
                    ind += 1
            if len(top_movies) == 0:
                estimate_rating = mean_rating
            else:
                sum_factor = 0
                for movie_i in top_movies:
                    estimate += self.train[user_index][movie_i] * top_movies[movie_i]
                    sum_factor += top_movies[movie_i]
                estimate_rating = estimate/sum_factor
        except ValueError:
            estimate_rating = mean_rating
        return estimate_rating

    def user_tests(self):
        cor = 0
        mse = 0
        self.svd()
        similarity = self.cal_similarity_user().astype('float16')
        #similarity = self.pearson().astype('float16')
        for _, row in self.test.iterrows():
            user = int(row['userId'])
            user_index = self.train_user_dict.index(user)
            movie = int(row['movieId'])
            ratings = row['rating']
            user_sim = np.argsort(-similarity[user_index])
            est = self.user_predict(user_index, movie, user_sim, similarity)
            #est = self.user_predict3(user_index, movie, user_sim, similarity)
            mse += (est - ratings) ** 2
            if abs(est - ratings) <= 0.3:
                cor += 1
        acc = cor/self.test.shape[0] * 100
        rmse = sqrt(mse/self.test.shape[0])
        print(f"****************     Acc: {acc}     **************")
        print(f"****************     RMSE: {rmse}     **************")

    def movie_tests(self):
        cor = 0
        mse = 0
        similarity = self.cal_similarity_movie()
        similarity = similarity.astype('float16')
        for _, row in self.test.iterrows():
            user = int(row['userId'])
            movie = int(row['movieId'])
            ratings = row['rating']
            user_index = self.train_user_dict.index(user)
            mean_rating = np.mean(self.train[user_index])
            try:
                movie_index = self.train_movie_dict.index(movie)  # since 'movie' is the movieId and we need to transform it from the real Id to the index in trainning set
                movie_sim = np.argsort(-similarity[movie_index])
                est = self.movie_predict(user_index, movie_index, similarity, movie_sim)
            except ValueError:
                est = mean_rating
            mse += (est - ratings) ** 2
            if abs(est - ratings) <= 0.3:
                cor += 1
        acc = cor / self.test.shape[0] * 100
        rmse = sqrt(mse / self.test.shape[0])
        print(f"****************     Acc: {acc}     **************")
        print(f"****************     RMSE: {rmse}     **************")

    def recommand(self, user_index):
        unseen = []
        top10 = []
        unseen_est = {}
        similarity = self.cal_similarity_user().astype('float16')
        user_sim = np.argsort(-similarity[user_index])
        for i in range(len(self.train_movie_dict)):
            if (self.train[user_index][i] == 0):
                unseen.append(self.train_movie_dict[i])
                est = self.user_predict(user_index, self.train_movie_dict[i], user_sim, similarity)
                unseen_est[self.train_movie_dict[i]] = est

        sorted_unseen = sorted(unseen_est.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(min(10, len(unseen))):
            top10.append(sorted_unseen[i][0])

        return top10

















