from fastapi.testclient import TestClient
from main import app
import pytest
from requests.auth import HTTPBasicAuth
import os

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    
def test_User():
    basic = HTTPBasicAuth(str(os.getenv('USER_API')), str(os.getenv('PASS_API')))
    response = client.get("/movie_reco_via_user",auth=basic)
    assert response.status_code == 200

def test_Movie():
    basic = HTTPBasicAuth(str(os.getenv('USER_API')), str(os.getenv('PASS_API')))
    response = client.get("/movie_reco_via_movie/25",auth=basic)
    assert response.status_code == 200
    
def test_New_User_noadimin():
    basic = HTTPBasicAuth(str(os.getenv('USER_API')), str(os.getenv('PASS_API')))
    payload = {'username': 12, 'password': 'senha123'}
    response = client.put("/new_user/",auth=basic,json=payload)
    assert response.status_code == 403
    
def test_New_rating():
    basic = HTTPBasicAuth(str(os.getenv('USER_API')), str(os.getenv('PASS_API')))
    payload = {'MovieID': 25, 'Rating': 5}
    response = client.put("/new_rating/",auth=basic, json=payload)
    assert response.status_code == 200
    
def test_New_stats_noadimin():
    basic = HTTPBasicAuth(str(os.getenv('USER_API')), str(os.getenv('PASS_API')))
    response = client.get("/stats/",auth=basic)
    assert response.status_code == 403
    
def test_New_validation_noadimin():
    basic = HTTPBasicAuth(str(os.getenv('USER_API')), str(os.getenv('PASS_API')))
    response = client.get("/new_validation/",auth=basic)
    assert response.status_code == 403