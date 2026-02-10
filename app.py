from flask import Flask, request
import joblib
import re

app = Flask(__name__)

# Load model + vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------------------------------
# Emotion keyword dictionaries
# -------------------------------
emotion_patterns = {
    "risk": [
        r"kill myself",
        r"end my life",
        r"want to die",
        r"suicide",
        r"no reason to live",
        r"give up on life",
        r"worthless",
        r"hopeless"
    ],
    "sadness": [
        r"\bsad\b",
        r"crying",
        r"depressed",
        r"miserable",
        r"down lately"
    ],
    "lonely": [
        r"lonely",
        r"alone",
        r"isolated",
        r"no one cares",
        r"nobody understands",
        r"by myself"
    ],
    "positive": [
        r"hopeful",
        r"getting better",
        r"stay strong",
        r"healing",
        r"improving"
    ]
}


emotion_colors = {
    "sadness":"#4da6ff",
    "lonely":"#bb86fc",
    "risk":"#ff4d4d",
    "positive":"#00e676"
}

# -------------------------------
# Clean text
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-zA-Z\s]","",text)
    return text.strip()

# -------------------------------
# Highlight function
# -------------------------------

def highlight_emotions(text):

    highlighted = text

    for emotion, patterns in emotion_patterns.items():
        color = emotion_colors[emotion]

        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)

            highlighted = regex.sub(
                lambda m: f'<span style="color:{color}; font-weight:bold; text-shadow:0 0 8px {color};">{m.group()}</span>',
                highlighted
            )

    return highlighted


# -------------------------------
# HTML UI
# -------------------------------

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>MindGuard AI</title>

<style>

body {
    margin:0;
    font-family:'Segoe UI',sans-serif;
    background: linear-gradient(270deg,#0f2027,#203a43,#2c5364);
    background-size:600% 600%;
    animation: gradientMove 12s ease infinite;
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
    color:white;
}

@keyframes gradientMove {
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}

.container {
    background:rgba(0,0,0,0.75);
    padding:35px;
    border-radius:20px;
    width:450px;
    text-align:center;
    backdrop-filter: blur(10px);
    box-shadow:0 0 40px rgba(0,255,255,0.2);
}

h1 {
    font-size:32px;
    font-weight:800;
    background:linear-gradient(90deg,#00f5ff,#00ff95);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

textarea {
    width:100%;
    height:120px;
    padding:12px;
    border-radius:12px;
    border:none;
    outline:none;
    resize:none;
    background:#111;
    color:white;
}

button {
    margin-top:15px;
    background:linear-gradient(90deg,#00f5ff,#00ff95);
    color:black;
    border:none;
    padding:12px 25px;
    border-radius:10px;
    font-weight:bold;
    cursor:pointer;
}

.result {
    margin-top:20px;
    font-size:22px;
    font-weight:bold;
}

.highlight-box {
    margin-top:15px;
    padding:12px;
    background:#111;
    border-radius:10px;
    font-size:14px;
    text-align:left;
}

</style>
</head>

<body>

<div class="container">

<h1>🧠 MindGuard</h1>
<p>Emotion-Aware Mental Health detector</p>

<form method="POST">
<textarea name="text" placeholder="Share your thoughts..." required></textarea>
<br>
<button type="submit">Analyze Mind</button>
</form>

{result}
{highlighted}

</div>
</body>
</html>
"""

# -------------------------------
# Flask route
# -------------------------------

@app.route("/", methods=["GET","POST"])
def home():

    result_html=""
    highlight_html=""

    if request.method=="POST":
        text=request.form["text"]

        cleaned=clean_text(text)
        vec=vectorizer.transform([cleaned])
        pred=model.predict(vec)[0]

        if pred==1:
            result_html='<div class="result" style="color:#ff4d4d;">HIGH RISK ⚠️</div>'
        else:
            result_html='<div class="result" style="color:#00e676;">LOW RISK ✅</div>'

        highlighted = highlight_emotions(text)

        highlight_html = f"""
        <div class="highlight-box">
        <b>Emotion Signals Detected:</b><br><br>
        {highlighted}
        </div>
        """

    page = HTML.replace("{result}",result_html)
    page = page.replace("{highlighted}",highlight_html)

    return page


if __name__=="__main__":
    app.run(debug=True)

