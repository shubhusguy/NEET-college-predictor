from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

artifact = joblib.load("neet_best_model.pkl")

model = artifact["model"]
model_name = artifact["model_name"]
target_encoder = artifact["target_encoder"]
category_frequency = artifact["category_frequency"]
rare_categories = set(artifact["rare_categories"])
metrics = artifact["metrics"]
n_classes = artifact["n_classes"]
trained_rows = artifact["trained_rows"]
min_air = artifact["min_air"]
max_air = artifact["max_air"]
category_options = artifact["category_options"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "model_name": model_name,
        "n_classes": n_classes,
        "trained_rows": trained_rows,
        "min_air": min_air,
        "max_air": max_air,
        "accuracy": metrics.get("Accuracy"),
        "top3_accuracy": metrics.get("Top-3 Accuracy"),
        "category_options": category_options
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    try:
        air = int(data["air"])
        gender = data["gender"]
        category = data["category"]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Missing or invalid fields: air, gender, category are required."}), 400

    if gender not in ("M", "F"):
        return jsonify({"error": "Gender must be 'M' or 'F'."}), 400

    gender_enc = 0 if gender == "M" else 1

    # Match the notebook's grouping exactly: rare categories -> 'OTHER'
    lookup_category = "OTHER" if category in rare_categories else category
    category_freq = category_frequency.get(lookup_category, 0)

    sample = pd.DataFrame({
        "AIR": [air],
        "Gender_enc": [gender_enc],
        "Category_freq_enc": [category_freq]
    })

    probabilities = model.predict_proba(sample)[0]
    top_n = min(3, len(probabilities))
    top_indices = probabilities.argsort()[::-1][:top_n]

    college_names = target_encoder.inverse_transform(top_indices)

    predictions = [
        {"college": str(college_names[i]), "probability": float(probabilities[idx])}
        for i, idx in enumerate(top_indices)
    ]

    out_of_range_warning = air < min_air or air > max_air

    return jsonify({
        "predictions": predictions,
        "model_used": model_name,
        "out_of_range_warning": out_of_range_warning
    })


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
