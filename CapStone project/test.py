import requests

img = {"image_path": "./data/test/beetroot/Image_1.jpg"}

url = "http://localhost:5000/predict"

print(requests.post(url, json=img).json())