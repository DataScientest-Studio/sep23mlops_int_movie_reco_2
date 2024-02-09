import pandas as pd
import pickle
from scipy.sparse import load_npz
import cloudpickle
import requests
import retrain_models


# load the data
def load_data():
    # read data from csv and feather
    ratings = pd.read_feather("./data/NewAproachModel_data.feather")
    ratings.movieId=ratings.movieId.astype(int)
    ratings=ratings.set_index('movieId')
    movies = pd.read_csv("./data/movies.csv")
    return ratings,movies


def load_model_data():
    movie_matrix =load_npz("./model/movie_matrix.npz")
    user_matrix =load_npz("./model/user_matrix.npz")
    
    # Load the saved model
    with open("./model/knn_movie.pkl", "rb") as f:
        knn_movie = cloudpickle.load(f)

    with open("./model/knn_user.pkl", "rb") as f:
        knn_user = cloudpickle.load(f)

    return movie_matrix,user_matrix,knn_movie,knn_user




def get_movie_recommendation_MovieId(movie_id_val,ratings,movies,movie_matrix,knn_movie):
    n_movies_to_reccomend = 5 
    try :        
        movie_idx = ratings.index.get_loc(movie_id_val)
        distances , indices = knn_movie.kneighbors(movie_matrix[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        list_indices=indices.squeeze().tolist()
        list_movieId=list(ratings.iloc[list_indices].index)
        list_movieId.remove(movie_id_val)
        df = movies[movies['movieId'].isin(list_movieId)].reset_index(drop=True)
        return df
    except :
        return "No movies found. Please check your input"




def get_movie_recommendation_UserID(user_id_val,ratings,movies,user_matrix,knn_user):
    n_users_to_compare = 2 
    try :        
        user_idx = ratings.columns.get_loc(str(user_id_val))
        distances , indices = knn_user.kneighbors(user_matrix[user_idx],n_neighbors=n_users_to_compare+1)   
        list_indices=indices.squeeze().tolist()
        df_compare=ratings.iloc[:,list_indices]
        df_compare.columns=['User','Closer_User','NCloser_User']
        df_out=df_compare[(df_compare.User==0)&(df_compare.Closer_User==5)]
        if df_out.shape[0]<5:
            df_out2=df_compare[(df_compare.User==0)&(df_compare.NCloser_User==5)]
            df_out=pd.concat([df_out,df_out2])
        list_movieId=df_out.iloc[0:5].index
        df = movies[movies['movieId'].isin(list_movieId)].reset_index(drop=True)
        return df
    except :
        return "No movies found. Please check your input"
    
    
def get_movie_contingency(user_id_val,ratings,top_movies,movies):
    try :        
        list_user = list(ratings[ratings[(str(user_id_val))]>0].index)
        new_list = [movie for movie in top_movies if movie not in list_user]
        new_list=new_list[:5]
        df = movies[movies['movieId'].isin(new_list)].reset_index(drop=True)
        return df
    except :
        return "No movies found. Please check your input"
    
async def trigger(ratings,movies):
    df_hr=pd.read_csv('./log_file/hist_rating.csv')
    if df_hr.shape[0]>=50:
        await trigger_retrain(ratings,movies)
        await trigger_validation(ratings,movies)
        df_hr.drop(df_hr.index,inplace=True)
        df_hr.to_csv('./log_file/hist_rating.csv')
    else:
        print('continue')
    return 200

async def trigger_retrain(ratings,movies):
    r= await retrain_models.train_model(ratings,movies)
    return r

async def trigger_validation(ratings,movies):
    r= await retrain_models.validate_model(ratings,movies)
    return r