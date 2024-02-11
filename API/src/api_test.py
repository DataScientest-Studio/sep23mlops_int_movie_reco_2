from fastapi.testclient import TestClient
from main import app
#from requests.auth import HTTPBasicAuth
#import pytest
#import warnings
#warnings.filterwarnings("ignore")


client = TestClient(app)

def health_root():
    response = client.get("/")
    assert response.status_code == 200