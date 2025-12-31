import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "rent_data.csv")
data = pd.read_csv(csv_path)

# Create category
def rent_category(rent):
    if rent < 7000:
        return "Cheap"
    elif rent < 15000:
        return "Moderate"
    else:
        return "Expensive"

data["rent_category"] = data["rent"].apply(rent_category)

# Split features
X_cat = data[["location", "house_type", "furnishing"]]
X_num = data[["size"]]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_cat_encoded = encoder.fit_transform(X_cat)

X = np.hstack((X_cat_encoded, X_num.values))

y_reg = data["rent"]
y_clf = data["rent_category"]

# Train models
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X, y_reg)

clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X, y_clf)

# Save files
with open(os.path.join(BASE_DIR, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

with open(os.path.join(BASE_DIR, "rent_regression_model.pkl"), "wb") as f:
    pickle.dump(reg_model, f)

with open(os.path.join(BASE_DIR, "rent_classification_model.pkl"), "wb") as f:
    pickle.dump(clf_model, f)

print("✅ TRAINING DONE. Encoder & models saved.")
