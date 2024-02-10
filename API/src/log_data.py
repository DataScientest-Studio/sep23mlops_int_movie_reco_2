import pandas as pd
import pickle
from scipy.sparse import csr_matrix,save_npz,load_npz
import cloudpickle
import math




def insert_new_user_log(user,userpass):
    temp=pd.DataFrame({"UserId":int(user),"password":userpass,"access":"user"},index=[0])
    login_name_pass = pd.read_csv("./data/login.csv")
    if login_name_pass[login_name_pass.UserId==int(user)].shape[0]>0:
        login_name_pass.loc[login_name_pass.UserId==int(user),'password']=str(userpass)
        login_name_pass.to_csv("./data/login.csv",index=False)
        return 200
    else:
        login_name_pass=pd.concat([temp,login_name_pass],ignore_index=True)
        login_name_pass.to_csv("./data/login.csv",index=False)
        return 200

def insert_new_user_rating(ratings,user):
    try:
        print(ratings[str(user)].shape[0])
        return 300
    
    except KeyError:
        ratings[str(user)]=0
        ratings.reset_index().to_feather('./data/NewAproachModel_data.feather')
        return 200
    
    except :
        return 400


def insert_new_movie_rating(movieID,UserID,rating,df_rating):
    try :
        df_rating.loc[[movieID],[str(UserID)]] = math.ceil(rating)
        df_rating.reset_index().to_feather('./data/NewAproachModel_data.feather')
        return 200
    except :
        return 400


def insert_new_movie(movieID,title,genres,df_rating,df_movieID):
    try :
        if df_movieID[df_movieID.movieId==movieID].shape[0]>0:
            return 400
        else:
            dict_mov={'movieId':[movieID],'title':[title],'genres':[genres]}
            df_movieID=pd.concat([df_movieID,pd.DataFrame.from_dict(dict_mov)],ignore_index=True)
            df_movieID.to_csv("./data/movies.csv")
            df_rating.loc[movieID]=0
            df_rating.reset_index().to_feather('./data/NewAproachModel_data.feather')
        return 200
    except :
        return 400


def prediction_log(df,user):
    df['user']=user
    try:
        df_hist=pd.read_csv('./log_file/hist_recommendation.csv')
        df_final=pd.concat([df_hist,df],ignore_index=False)
        df_final.drop_duplicates(inplace=True)
        df_final.to_csv('./log_file/hist_recommendation.csv',index=False)
    except:
        df.to_csv('./log_file/hist_recommendation.csv',index=False) 
    return 200
    
def rating_log(user,movieID,rating):
    dict_mov={'user':[user],'movieID':[movieID],'rating':[rating]}
    df=pd.DataFrame.from_dict(dict_mov)
    try:
        df_hist=pd.read_csv('./log_file/hist_rating.csv')
        df_final=pd.concat([df_hist,df],ignore_index=False)
        df_final.drop_duplicates(inplace=True)
        df_final.to_csv('./log_file/hist_rating.csv',index=False)
    except:
        df.to_csv('./log_file/hist_rating.csv',index=False) 
    return 200



def get_stats():
    try:
        df_s=pd.read_csv('./model/log_stats.csv').tail(1)
        df_hr=pd.read_csv('./log_file/hist_rating.csv')
        df_hrec=pd.read_csv('./log_file/hist_recommendation.csv')
        df_m=df_m.tail(1)
        df_s=df_s.tail(1)
        
        date_ret=df_s.Date.iloc[0]
        
        acuracy_ret=df_s.Acuracy.iloc[0]
        
        df_hr=df_hr.rename(columns={'movieID':'movieId'})
        df_join=df_hrec.merge(df_hr)
        df_join.loc[df_join.rating<=3,'rating']=1
        df_join.loc[df_join.rating>3,'rating']=5
        df_join=df_join.groupby(['rating'],as_index=False)['movieId'].count()
        df_join['index_n']=0
        df_join=pd.pivot(df_join,index='index_n',columns='rating',values='movieId')
        test_val=df_join.reset_index(drop=True)
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

        live_acuracy=test_val.Acuracy.iloc[0]
        
        return pd.DataFrame.from_dict({'LastModelValidation':[date_ret],'ModelStatsTrainingAcuracy':[acuracy_ret],'ModelStatsLiveAcuracy':[live_acuracy]}),200
    except:
        return pd.DataFrame.from_dict({'LastModelValidation':['Not Available'],'ModelStatsTrainingAcuracy':[0],'ModelStatsLiveAcuracy':[0]}),400
        
        
