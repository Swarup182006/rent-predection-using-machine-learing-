from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
...
app = Flask(__name__)
CORS(app)


# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

reg_model = pickle.load(open(os.path.join(MODEL_DIR, "rent_regression_model.pkl"), "rb"))
clf_model = pickle.load(open(os.path.join(MODEL_DIR, "rent_classification_model.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb"))

@app.route("/")
def home():
    return "RentVision Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Get user input
    location = data["location"]
    house_type = data["house_type"]
    furnishing = data["furnishing"]
    size = data["size"]

    # Encode categorical
    cat_data = [[location, house_type, furnishing]]
    cat_encoded = encoder.transform(cat_data)

    # Combine with size
    final_input = np.hstack((cat_encoded, [[size]]))

    rent = int(reg_model.predict(final_input)[0])
    category = clf_model.predict(final_input)[0]

    return jsonify({
        "predicted_rent": rent,
        "category": category
    })
@app.route("/predict-multiple", methods=["POST"])
def predict_multiple():
    houses = request.json  # list of houses
    results = []

    for house in houses:
        location = house["location"]
        house_type = house["house_type"]
        furnishing = house["furnishing"]
        size = house["size"]

        cat_input = [[location, house_type, furnishing]]
        cat_encoded = encoder.transform(cat_input)

        final_input = np.hstack((cat_encoded, [[size]]))

        rent = int(reg_model.predict(final_input)[0])
        category = clf_model.predict(final_input)[0]

        results.append({
            "location": location,
            "house_type": house_type,
            "predicted_rent": rent,
            "category": category
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
