from fastapi.testclient import TestClient
from main import app
from requests.auth import HTTPBasicAuth
import warnings
warnings.filterwarnings("ignore")



client = TestClient(app)

basic = HTTPBasicAuth(11, 'senha123')


def health_root():
    response = client.get("/")
    assert response.status_code == 200
    
def User_pred_root():
    response = client.get("/movie_reco_via_user",auth=basic)
    assert response.status_code == 200