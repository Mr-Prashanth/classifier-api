from flask import Flask
import text_classifier as tc

app = Flask(__name__)

@app.route('/<text>')
def text(text):
    return tc.classify(text)
@app.route('/<text>/<float:thershold>')
def text_thershold(text,thershold):
    return tc.classify(text, thershold)
if __name__ == "__main__":
    app.run()