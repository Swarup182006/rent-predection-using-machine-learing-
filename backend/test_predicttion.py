import requests

url = "http://127.0.0.1:5000/predict-multiple"

data = [
    {
        "location": "JJ",
        "house_type": "2BHK",
        "furnishing": "Semi",
        "size": 780
    },
    {
        "location": "BN",
        "house_type": "1BHK",
        "furnishing": "Unfurnished",
        "size": 500
    },
    {
        "location": "Old Gunupur",
        "house_type": "3BHK",
        "furnishing": "Furnished",
        "size": 1150
    }
]

response = requests.post(url, json=data)
print(response.json())
