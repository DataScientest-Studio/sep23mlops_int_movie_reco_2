import pandas as pd
from scipy.sparse import csr_matrix,save_npz,load_npz
from sklearn.neighbors import NearestNeighbors
import math
import cloudpickle
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")



async def validate_model(ratings,movies):
    # read data bases
    #movies = pd.read_csv("./data/movies.csv")
    #ratings = pd.read_feather("./data/NewAproachModel_data.feather")
    #ratings.movieId=ratings.movieId.astype(int)
    #ratings=ratings.set_index('movieId')


    # Transform data base
    ratings_user=ratings.sample(frac=0.5)
    ratings_user.index=ratings_user.index.astype(str)
    ratings_user=ratings_user.T
    ratings_user=ratings_user.rename_axis('UserId', axis=1)
    ratings_user.index=ratings_user.index.astype(int)

    # Hide some good and bad ratings

    # Select random user and movies to be hidden
    X,X_test=train_test_split(ratings_user,test_size=0.01, random_state=42)
    csr_ratings_user = csr_matrix(X_test)

    df_final_test=[]
    for a in range(X_test.shape[0]):
        try:
            lista_movies=csr_ratings_user[a].indices
            lista_ratings=csr_ratings_user[a].data
            df_test=pd.DataFrame.from_dict({'movie_idx':lista_movies,'ratings':lista_ratings})
            df_test['user']=X_test.iloc[a].name
            df_final_test=pd.concat([df_final_test,df_test],ignore_index=True)
        except:
            lista_movies=csr_ratings_user[a].indices
            lista_ratings=csr_ratings_user[a].data
            df_test=pd.DataFrame.from_dict({'movie_idx':lista_movies,'ratings':lista_ratings})
            df_test['user']=X_test.iloc[a].name
            df_final_test=df_test
        
        
    df_final_test=df_final_test[(df_final_test.ratings==5)|(df_final_test.ratings==1)]

    # Replace the rating for 0.
    for a in range(df_final_test.shape[0]):
        user=int(df_final_test.iloc[a]['user'])
        movie_idx=int(df_final_test.iloc[a]['movie_idx'])
        movie=X_test.columns[movie_idx]
        X_test.loc[user][movie]=0
    
    # Find the movieId, the columns was the index only
    df_final_test['movieId']='0'
    for a in range(df_final_test.shape[0]):
        df_final_test['movieId'].iloc[a]=X_test.columns[df_final_test.iloc[a]['movie_idx']]

    #Train model with the new database hidding some movies evaluation

    df_train=pd.concat([X,X_test])
    csr_ratings_user = csr_matrix(df_train)
    knn_user = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn_user.fit(csr_ratings_user)

    #Define function to recommend movie
    def get_movie_recommendation_UserID(user_id_val):
        n_users_to_compare = 1 
        if 1==1:      
            user_idx = df_train.index.get_loc(user_id_val)
            distances , indices = knn_user.kneighbors(csr_ratings_user[user_idx],n_neighbors=n_users_to_compare+1)    
            list_indices=indices.squeeze().tolist()
            df_compare=ratings.iloc[:,list_indices]
            df_compare.columns=['User','Closer_User']
            df_out=df_compare[(df_compare.User==0)&(df_compare.Closer_User==5)]
            list_movieId=df_out.index
            df = movies[movies['movieId'].isin(list_movieId)].reset_index(drop=True)
        return df



    
    # Run recommendation

    usertest=df_final_test.user.unique()
    df_predict=[]
    for a in usertest:
        try:
            df_temp=get_movie_recommendation_UserID(a)
            df_temp['User']=a
            df_predict=pd.concat([df_predict,df_temp])
        except:
            df_temp=get_movie_recommendation_UserID(a)
            df_temp['User']=a
            df_predict=df_temp
        
    # Join real evaluation with recommendetion one

    df_final_test.movieId=df_final_test.movieId.astype(int)
    df_final_test=df_final_test.rename(columns={'user':'User'})

    test_val=df_final_test.merge(df_predict,how='left')
    test_val=test_val[~test_val.title.isnull()]
    test_val=test_val.groupby(['ratings'],as_index=False)['movieId'].count()
    test_val['index_n']=0
    test_val=pd.pivot(test_val,index='index_n',columns='ratings',values='movieId')
    test_val=test_val.reset_index(drop=True)
    test_val=test_val.rename(columns={1.0:'False_Positive',5.0:'True_Positive'})
    try:
        test_val.True_Positive[0]
    except:
        test_val['True_Positive']=0
    try:
        test_val.False_Positive[0]
    except:
        test_val['False_Positive']=0
    test_val['Acuracy']=100*(test_val.True_Positive/(test_val.True_Positive+test_val.False_Positive))
    test_val['Date']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        df_stats=pd.read_csv('./model/log_stats.csv')
        df_stats=pd.concat([df_stats,test_val],ignore_index=True)
        df_stats.to_csv('./model/log_stats.csv',index=False)
    except:
        test_val.to_csv('./model/log_stats.csv',index=False)

    return 200




async def train_model(ratings,movies):
    # read data bases
    #movies = pd.read_csv("./data/movies.csv")
    #ratings = pd.read_feather("./data/NewAproachModel_data.feather")
    #ratings.movieId=ratings.movieId.astype(int)
    #ratings=ratings.set_index('movieId')


    # Train model to reccomend a movie based on some other movie

    csr_ratings = csr_matrix(ratings)
    save_npz("./model/movie_matrix.npz", csr_ratings)
    knn_movie = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn_movie.fit(csr_ratings)

    with open("./model/knn_movie.pkl", "wb") as f:
        cloudpickle.dump(knn_movie, f)


    # Train model to reccomend a movie based on User
    ratings_user=ratings.copy()
    ratings_user.index=ratings_user.index.astype(str)
    ratings_user=ratings_user.T
    ratings_user=ratings_user.rename_axis('UserId', axis=1)
    ratings_user.index=ratings_user.index.astype(int)
    csr_ratings_user = csr_matrix(ratings_user)
    save_npz("./model/user_matrix.npz", csr_ratings)
    knn_user = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn_user.fit(csr_ratings_user)
    with open("./model/knn_user.pkl", "wb") as f:
        cloudpickle.dump(knn_user, f)
    
    test_val=pd.DataFrame.from_dict({'date':[datetime.now().strftime("%Y-%m-%d %H:%M:%S")],'comment':'model retreined'})
    
    try:
        df_stats=pd.read_csv('./model/log_model.csv')
        df_stats=pd.concat([df_stats,test_val],ignore_index=True)
        df_stats.to_csv('./model/log_model.csv',index=False)
    except:
        test_val.to_csv('./model/log_model.csv',index=False)
    
    return 200