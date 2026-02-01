import os
print(os.getcwd())
from flask import Flask, request, render_template_string
from model.resume_model import predict

app = Flask(__name__)

HTML = """
<h2>AI Resume Screening</h2>
<form method="post">
<textarea name="resume" rows="10" cols="60"></textarea><br><br>
<button type="submit">Predict</button>
</form>
<h3>{{result}}</h3>
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = ""
    if request.method == "POST":
        result = "Predicted Role: " + predict(request.form["resume"])
    return render_template_string(HTML, result=result)

app.run(debug=True)
