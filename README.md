Recommender system using collaborative filtering
============================================================================

Overview
-----------------------------------------------------------------------------
This project generates Recommender system by collaborative filtering methods. It can be divided into Memory-based Collaborative Filtering and Model-based Collaborative Filtering. In Memory-based Collaborative Filtering The recommendation results are based on the similarity of between users and between movies. To calculate the similarity, we used cosine distance. Otherwise, for the Model-based Collaborative Filtering the used model is Matrix Factorization. The idea behind such models is that attitudes or preferences of a user can be determined by a small number of hidden latent factors. These factors are also called Embeddings, which represent different characteristics for users and items. 

<!-- ???Due to the limitation of the data source of user and item fratures, 
???From an comprehensive view we conclude that our theory and algorithms are ???feasible while at the same time there are still some aspects expected to improve.
???补充成果和提升空间。
 -->
<!---Theories
The algorithm we used is based on Collaborative Filtering algorithm, it contains memory-based Collaborative Filtering and Model-based Collaborative Filtering. 
--->
The memory-based algorithm contains following steps:

Data Preparation including train_test_split
Transform from DataFrame to matrix of users and matrix of movies
Calculate the user-user similarity matrix and movie-movie similarity by using Sklearn.pairwise
Make prediction for the ratings of each user with regard to each movie
Sort the ratings to provide top 10 movies for each user.
<!-- The flaw of CF algorithm is that, when users have few preferences, the preference matrix would become sparse, which will affect the accuracy of similarity. how to improve the accuracy ？
 -->

<!-- We create a pseudo user-ratings vector for every user u in database, which consists of the item ratings provided by the user u, where available, and those predicted by the content-based predictor otherwise.
provide more details!!! -->
The model-based algorithm contains following steps:

Data Preparation including train_test_split
Train the model with bias using SGD
Calculate the weighting matrix and its biases
Predict the ratings based on the weighting matrix and biases obtained from training

Goals
-----------------------------------------------------------------------------
The goals for this project are:
(1) Implement memory-based(user-based, item-based) and model-based(matrix factorization) for recommender system.
(2) Compare the RMSE for both algorithms and find a better one.
(3) Do recommendation in real world.

Environment Settings
-----------------------------------------------------------------------------
- Python 3.6
<!-- In this model-based Collaborative Filtering, we used biased and unbiased training methods.  -->

<!-- Functions
-----------------------------------------------------------------------------
dataset.py
memory_based.py
matrix_factorization.py
recommender.py -->

Instruction
-----------------------------------------------------------------------------
To test the accuracy and RMSE for model-based algorithm:
	For 100k:
	python3.6 recommender.py 100k model test
	For 1m:
	python3.6 recommender.py 1m model test
To test the accuracy and RMSE for memory-based algorithm:
	For 100k:
	python3.6 recommender.py 100k memory test
	For 1m:
	python3.6 recommender.py 1m memory test
To make recommendation for specific user(user_name) using memory based algorithm:
	For 100k:
	python3.6 recommender.py 100k memory user_name
	For 1m:
	python3.6 recommender.py 1m memory user_name
To make recommendation for specific user(user_name) using model based algorithm:
	For 100k:
	python3.6 recommender.py 100k model user_name
	For 1m:
	python3.6 recommender.py 1m model user_name

Output
-----------------------------------------------------------------------------
For test:
To test the accuracy and RMSE, there will be instructions for 

Dataset
-----------------------------------------------------------------------------
The large dataset we used contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.
The small dataset from [MovieLens](http://movielens.org) contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

