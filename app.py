from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('pipe_line_model.h5')

@app.route('/')
def index():
    return "Welcome To Arabic sentment analysis"

@app.route('/predict/',  methods=['GET'])
def predict():
    client = request.args['text']
    
    if model.predict([client])[0] == 0:
        prediction = "neutral"
    elif model.predict([client])[0] == 1:
        prediction = "negative"
    else:
        prediction = "positive"
    return jsonify(Prediction=str(prediction))

    
if __name__ == '__main__':
    app.run(debug=True)
