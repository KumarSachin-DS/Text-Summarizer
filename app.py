# backend.py
# pip install flask newspaper3k transformers torch flask-cors

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from newspaper import Article
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load summarizer once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

def fetch_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def generate_summary(text, num_sentences=3):
    min_length = num_sentences * 20
    max_length = num_sentences * 50
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

@app.route("/")
def index():
    with open("index.html", "r") as f:
        return f.read()

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    try:
        article_text = fetch_article_text(url)
        summary = generate_summary(article_text, 3)
        return jsonify({"summary": summary, "article_snippet": article_text[:500]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
