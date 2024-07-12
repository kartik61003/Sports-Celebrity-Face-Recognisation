from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import util

app = Flask(__name__)
cors = CORS(app)
@app.route('/classify_image', methods = ['GET','POST'])
@cross_origin()
def classify_image():
    image_data = request.form['image_data']
    response = jsonify(util.classify_image(image_data))
    return response

if(__name__=="__main__"):
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)
    