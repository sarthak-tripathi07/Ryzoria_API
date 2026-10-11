from flask import Flask, request, jsonify
import pickle
from urllib.parse import urlparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  #
# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Trusted domains
trusted_domains = {
    "google.com",
    "www.google.com",
    "github.com",
    "www.github.com",
    "facebook.com",
    "www.facebook.com",
    "wikipedia.org",
    "www.wikipedia.org"
}

@app.route("/")
def home():
    return "Ryzoria API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    domain = urlparse(url).netloc.lower()

    # Trusted website
    if domain in trusted_domains:
        prediction = "Good"
    else:
        X = vectorizer.transform([url])
        result = model.predict(X)[0]
        prediction = str(result)

    return jsonify({
        "url": url,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)