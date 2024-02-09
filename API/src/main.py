import pandas as pd
from fastapi import FastAPI
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from pydantic import BaseModel
from typing import List,Optional
import os
import time


time.sleep(30)

# My own Library
from predict import load_data,load_model_data,get_movie_recommendation_MovieId,get_movie_recommendation_UserID,get_movie_contingency,trigger,trigger_validation
from log_data import insert_new_user_log,insert_new_user_rating,insert_new_movie_rating,insert_new_movie,prediction_log,rating_log,get_stats



class User(BaseModel):
    username: int
    password: str

class MovieRecommendation(BaseModel):
    Order: int
    MovieID: int
    MovieName: str

class MovieRating(BaseModel):
    MovieID: int
    Rating: int

class outputAPI(BaseModel):
    Recommendation: List[MovieRecommendation]
    
class NewMovie(BaseModel):
    MovieID: int
    MovieName: str
    MovieGenres: str
    
class MovieStats(BaseModel):
    LastModelTraining: str
    ModelStatsTrainingAcuracy: int
    ModelStatsLiveAcuracy: int


security=HTTPBasic()

app = FastAPI(
    title="Movie Recommender API",
    description="API build up by Guillherme powered by FastAPI.",
    version="1.0.1",
    openapi_tags=[
        {
            'name': 'home',
            'description': 'entry of the API'
        },
        {
            'name': 'movie_recommendation_via_user',
            'description': 'put the userId and get Top 5 recommended movies'
        },
        {
            'name': 'movie_recomendation_via_movie',
            'description': 'put the movieId and get Top 5 recommended movies'
        }
    ]
)


app.ratings,app.movies=load_data()

app.movie_matrix,app.user_matrix,app.knn_movie,app.knn_user=load_model_data()

app.top_movies=app.ratings.sum(axis=1).sort_values(ascending=False).index

#Authentication

@app.get("/auth", summary="Authentication", include_in_schema=False)
def Auth(credentials: HTTPBasicCredentials=Depends(security) ):
    # Credentials

    ##
    app.username=int(credentials.username)
    password=credentials.password
    login_name_pass = pd.read_csv("./data/login.csv")
    
    
    # Check User
    try:
        check_pass=login_name_pass[login_name_pass.UserId==app.username].password.iloc[0]
    except KeyError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid User')
    
    #Check User password
    
    if not(check_pass==password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid Password')
    else:
        app.access=login_name_pass[login_name_pass.UserId==app.username].access.iloc[0]
    return 'authenticated'


# API Root
@app.get('/', tags= ["home"])
def get_index():
    """Returns greetings
    """
    return {'greetings': 'welcome, test 05/02/2024'}
    
@app.get("/movie_recommendation_via_user/", tags=["movie_recommendation_via_user"],response_model=outputAPI)
def get_movie_from_user( auth: str = Depends(Auth)):
    """
    In this endpoint the user will put the userId into the API and will return 5 movies
    which are most likely similiar rated by users therefore in the same cluster
    :param input_user:
    :return:
    """
    input_user = int(app.username)
    df = get_movie_recommendation_UserID(input_user,app.ratings,app.movies,app.user_matrix,app.knn_user)
    dic_reco=[]
    try :
        for a in range(df.shape[0]):
            temp={"Order":a+1,"MovieID":df.movieId[a],"MovieName":str(df.title[a])}
            dic_reco.append(temp)
    except :
        # Run contingency code - Select the most view and better ratted movies, just sum all
        df = get_movie_contingency(input_user,app.ratings,app.top_movies,app.movies)
        for a in range(df.shape[0]):
            temp={"Order":a+1,"MovieID":df.movieId[a],"MovieName":str(df.title[a])}
            dic_reco.append(temp)
    prediction_log(df,input_user)
    return outputAPI(Recommendation=dic_reco)
    
    
@app.get("/movie_recomendation_via_movie/{input_movie}", tags=["movie_recomendation_via_movie"])
def get_movie_from_movie(input_movie: int, auth: str = Depends(Auth)):
    """
    In this endpoint the user will put the movieId into the API and will return 5 movies
    which are similiar rated and therefore in the same cluster
    :param input_movie:
    :return:
    """
    input_movie = int(input_movie)
    df = get_movie_recommendation_MovieId(input_movie,app.ratings,app.movies,app.movie_matrix,app.knn_movie)
    dic_reco=[]
    try :
        for a in range(df.shape[0]):
            temp={"Order":a+1,"MovieID":df.movieId[a],"MovieName":str(df.title[a])}
            dic_reco.append(temp)
    except :
        temp={"Order":0,"MovieID":0,"MovieName":"NotFound"}
        dic_reco.append(temp)

    return outputAPI(Recommendation=dic_reco)

@app.put("/new_user/", name='register new user', tags=["admin tasks"])
def set_new_user(user: User, auth: str = Depends(Auth)):
    if app.access=="admin":
        status=insert_new_user_log(user.username,user.password)
        if status == 200:
            status=insert_new_user_rating(app.ratings,user.username)
            if status==200:
                app.ratings,app.movies=load_data()
                raise HTTPException(status_code=200, detail="New User Created !!")
            elif status==300:
                raise HTTPException(status_code=200, detail=f"Changed password from user {user.username} !!")
            else:
                raise HTTPException(status_code=400, detail="Not able to insert User to rating")
        else:
            raise HTTPException(status_code=400, detail="Not able to insert User to login base")
    else:
        raise HTTPException(status_code=400, detail="Not sufficient rights")
    
@app.put("/new_rating/", name='register new rating')
async def set_new_rating(rating:MovieRating, auth: str = Depends(Auth)):
    # 
    status=insert_new_movie_rating(rating.MovieID,app.username,rating.Rating,app.ratings)
    if status==200:
        app.ratings,app.movies=load_data()
        rating_log(app.username,rating.MovieID,rating.Rating)
        await trigger(app.ratings,app.movies)
        raise HTTPException(status_code=200, detail="New rating Created !!")
    else:
        raise HTTPException(status_code=400, detail="User or movie not found")
    

@app.put("/new_movie/", name='register new user', tags=["admin tasks"])
def set_new_movie(newmovie: NewMovie, auth: str = Depends(Auth)):
    if app.access=="admin":
        status=insert_new_movie(newmovie.MovieID,newmovie.MovieName,newmovie.MovieGenres,app.ratings,app.movies)
        if status == 200:
            raise HTTPException(status_code=200, detail="New Movie Created !!")
        else:
            raise HTTPException(status_code=400, detail="Not able to insert New Movie to database")
    else:
        raise HTTPException(status_code=400, detail="Not sufficient rights")
    
    
@app.get("/stats/", name='get model stats', tags=["admin tasks"],response_model=MovieStats)
def movie_stats(auth: str = Depends(Auth)):
    if app.access=="admin":
        status_return,status=get_stats()
        if status == 400:
            raise HTTPException(status_code=400, detail="Not able to access stats, model must be retrained at least once")
    else:
        raise HTTPException(status_code=400, detail="Not sufficient rights")
    return MovieStats(LastModelTraining=status_return.LastModelTraining.iloc[0],ModelStatsTrainingAcuracy=status_return.ModelStatsTrainingAcuracy.iloc[0],ModelStatsLiveAcuracy=status_return.ModelStatsLiveAcuracy.iloc[0])
    
    
@app.get("/validade_model/", name='get model stats', tags=["admin tasks"],response_model=MovieStats)
async def validade_model(auth: str = Depends(Auth)):
    if app.access=="admin":
        r = await trigger_validation(app.ratings,app.movies)
        if status == 400:
            raise HTTPException(status_code=400, detail="Not able to avalidate model")
    else:
        raise HTTPException(status_code=400, detail="Not sufficient rights")
    return MovieStats(LastModelTraining=status_return.LastModelTraining.iloc[0],ModelStatsTrainingAcuracy=status_return.ModelStatsTrainingAcuracy.iloc[0],ModelStatsLiveAcuracy=status_return.ModelStatsLiveAcuracy.iloc[0])
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)