from fastapi.testclient import TestClient
from main import app
from requests.auth import HTTPBasicAuth


client = TestClient(app)

basic = HTTPBasicAuth('11', 'senha123')


def health_root():
    response = client.get("/")
    assert response.status_code == 200
    
def User_pred_root():
    response = client.get("/movie_reco_via_user")
    assert response.status_code == 200