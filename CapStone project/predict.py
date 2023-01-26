import numpy as np
import matplotlib.pyplot as plt
import tensorflow.lite as tflite
from flask import Flask, jsonify, request
from skimage.transform import resize

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    
    interpreter = tflite.Interpreter(model_path='./fruits_vegetables_MobileNet.tflite')
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
        
    client = request.get_json()

    img = plt.imread(client["image_path"])
    img = resize(img, (150, 150))
    X = img
    X = X.reshape(1, 150, 150, 3)
    X = np.float32(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    
    result = {"Vegetable or fruit": float(preds[0][0])}
    
    return jsonify(result)

if __name__ == "__main__":
    
    app.run(debug=True, host="0.0.0.0", port=5000)
