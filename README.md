# movie_recommendation_system_DS

Collaborative recommender system
--------------------------------
Collaborative filtering is based on the assumption that people who agreed in the past will agree in the future, and that they will like similar kinds of items as they liked in the past. The system generates recommendations using only information about rating profiles for different users or items. By locating peer users/items with a rating history similar to the current user or item, they generate recommendations using this neighborhood. This approach builds a model from a user’s past behaviors (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that the user may have an interest in. Collaborative filtering methods are classified as memory-based and model-based.

Success Criteria
--------------------------------
Success will be measured by implementing collaborative and content-based models that can return movie recommendations to a user. The goal is to provide reviews that we find sensible based on either reviews that the user enters, or based on a film given to the content-based system. A good recommendation algorithm can be extremely useful for streaming companies, as a constant stream of accurate or interesting recommendations will keep users engaged with the platform.

Members
--------------------------------
|         Name             
|--------------------------
       
|Guilherme Santos       
        

About DataSet
------------


•	movie.csv ->
movieId,title,genres

•	ratings.csv ->
userId, movieId, rating, timestamp



API Set-up (API/src/main.py)
-------------------
start the API via Anaconda prompt
```python main.py```
do not forget to comment the line 29 and uncomment the line 32 of the main.py

API call example (main.py)
-------------------
•	to check it 

```http://localhost:5000/```


