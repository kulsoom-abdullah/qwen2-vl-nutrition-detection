import requests
import sys
import os

# Point to your local FastAPI server
URL = "http://127.0.0.1:8000/predict"

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python test_api_local.py <path_to_image>")
        print("Using default placeholder 'test_image.jpg' if it exists...")
        image_path = "test_image.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    print(f"Sending {image_path} to {URL}...")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(URL, files=files)

        print(f"Status: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except Exception:
            print(f"Raw Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {URL}. Is the server running?")

if __name__ == "__main__":
    main()
