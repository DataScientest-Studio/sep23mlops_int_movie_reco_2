import pandas as pd
import numpy as np
import cloudpickle
from scipy.sparse import csr_matrix,save_npz,load_npz
import os


def load_data():
    # read data from csv and feather
    ratings = pd.read_feather("./raw_data/NewAproachModel_data.feather")
    ratings.movieId=ratings.movieId.astype(int)
    ratings=ratings.set_index('movieId')
    movies = pd.read_csv("./raw_data/movies.csv")
    login_name_pass = pd.read_csv("./raw_data/login.csv")
    return ratings,movies,login_name_pass


def load_model_data():
    movie_matrix =load_npz("./raw_data/movie_matrix.npz")
    user_matrix =load_npz("./raw_data/user_matrix.npz")
    
    # Load the saved model
    with open("./raw_data/knn_movie.pkl", "rb") as f:
        knn_movie = cloudpickle.load(f)

    with open("./raw_data/knn_user.pkl", "rb") as f:
        knn_user = cloudpickle.load(f)

    return movie_matrix,user_matrix,knn_movie,knn_user


def save_data(ratings,movies,login_name_pass):
    # read data from csv and feather
    ratings.reset_index().to_feather('./data/NewAproachModel_data.feather')
    movies.to_csv("./data/movies.csv")
    login_name_pass.to_csv("./data/login.csv",index=False)
    return 200

def save_model_data(movie_matrix,user_matrix,knn_movie,knn_user):
    
    save_npz("./model/movie_matrix.npz", movie_matrix)
    save_npz("./model/user_matrix.npz", user_matrix)
    
    with open("./model/knn_movie.pkl", "wb") as f:
        cloudpickle.dump(knn_movie, f)
    
    with open("./model/knn_user.pkl", "wb") as f:
        cloudpickle.dump(knn_user, f)

    return 200




if (os.environ.get('REFRESH_DATA')=='True') or (len(os.listdir('./data'))==0):
    ratings,movies,login_name_pass=load_data()
    movie_matrix,user_matrix,knn_movie,knn_user=load_model_data()
    save_data(ratings,movies,login_name_pass)
    save_model_data(movie_matrix,user_matrix,knn_movie,knn_user)
    print('Data updated in volume')
else:
    print('Data already in volume')