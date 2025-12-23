from fastapi.testclient import TestClient
from nutrition_detector.api.app import app
import io
from PIL import Image

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_mock():
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    
    assert response.status_code == 200
    json_response = response.json()
    assert "boxes" in json_response
    assert "raw_text" in json_response
    # Mock engine returns "nutrition-table<box(100,100),(200,200)>"
    # 100/1000 = 0.1, 200/1000 = 0.2
    assert json_response["boxes"] == [[0.1, 0.1, 0.2, 0.2]]
