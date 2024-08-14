import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

import sklearn
print('sklearn',sklearn.__version__) 

# Load the model
model = pickle.load(open('model.pkl', 'rb'))
print(type(model))
# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# API endpoint
@app.route('/glasses-identification', methods=['POST'])
def glasses_identification():
    # Get the data from the POST request
    data = request.get_json(force=True)
    print("Received data:", data)
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}, Type: {type(value)}")

    # Extract feature values from the JSON data and convert to a list of floats
    features = list(data.values())
    features = [float(value) for value in features]
    print("Extracted features:", features)

    # Convert features to NumPy array
    #X_test = np.array([features])
    try:
        # Make prediction using the model
    #    y_test_hat = model.predict(X_test)
        y_test_hat = model.predict([features])
        # Take the first value of prediction
        output = int(y_test_hat[0])

        print("Prediction:", y_test_hat)
     # Return the prediction
        return jsonify({'prediction': output})
    except Exception as e:
        print(e)  # Print the exception here
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
