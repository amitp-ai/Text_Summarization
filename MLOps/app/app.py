from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok #for colab only

app = Flask(__name__)
run_with_ngrok(app) #for colab only
# model
# loggger and csvlogger
# wandb

@app.route("/")
def index():
    """Provide simple health check route"""
    return "Hello, World!"

@app.route("/v1/summarize/<int:desc>", methods=["GET", "POST"])
def summarize():
    """Summarize the input text"""
    if request.method == "GET":
        # res = request.args.get('description')
        res = desc
    elif request.method == "POST":
        res = request.get_json()
    res = res + ' - hello world!'
    return jsonify({"summary": res})


def main():
    """Run the app"""
    # app.run(host='0.0.0.0', port=8080)
    app.run() #debug=True)

if __name__ == '__main__':
    main()
