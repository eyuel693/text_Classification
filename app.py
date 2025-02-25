from flask import Flask, render_template, request
from text_classification import classify_text  # Import the classification function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    classification = None
    accuracy = None
    similarities = None

    if request.method == "POST":
        query = request.form.get("query")
        if query:
            classification, accuracy, similarities = classify_text(query)

    return render_template("index.html", classification=classification, accuracy=accuracy, similarities=similarities)


if __name__ == "__main__":
    app.run(debug=True)
