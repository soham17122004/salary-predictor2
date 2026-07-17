from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/salary_prediction_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")
category_options = joblib.load("model/category_options.pkl")
numeric_ranges = joblib.load("model/numeric_ranges.pkl")

categorical_cols = list(category_options.keys())
numeric_cols = list(numeric_ranges.keys())


@app.route("/")
def home():
    return render_template(
        "index.html",
        category_options=category_options,
        numeric_ranges=numeric_ranges,
        prediction=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Build a single-row dataframe from the form, matching training format
        raw_input = {}

        for col in numeric_cols:
            raw_input[col] = [float(request.form[col])]

        for col in categorical_cols:
            raw_input[col] = [request.form[col]]

        input_df = pd.DataFrame(raw_input)

        # One-hot encode the same way training did
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align columns exactly to what the model was trained on
        input_final = input_encoded.reindex(columns=feature_names, fill_value=0)

        input_scaled = scaler.transform(input_final)

        pred = model.predict(input_scaled)[0]
        prediction = round(float(pred), 2)
        error = None

    except Exception as e:
        prediction = None
        error = str(e)

    return render_template(
        "index.html",
        category_options=category_options,
        numeric_ranges=numeric_ranges,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
