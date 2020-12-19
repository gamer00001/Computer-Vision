import cv2
import json

import tensorflow as tf
import numpy as np
import yolo_object_detection

from flask import Flask, request, Response
import uuid


def prepare(file):
        IMG_SIZE = 28
        img_array = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        img_array = np.array(img_array) / 255.0
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)




def signPredictor(array):
    model = tf.keras.models.load_model(r"C:\Users\hira\PycharmProjects\SignDetection\detectSign.model")
    Categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Alif", "Bay", "Pay", "Tay", "Taay", "Say",
                  "Chay", "Hay", "Khay", "Dal", "Daal", "Zaal", "Ray", "Zay", "Saay", "Seen", "Saad",
                  "Daad", "Zuay", "Aeen", "Fay", "Qaaf", "Kaaf", "Gahf", "Laam", "noonGunna", "Wao",
                  "Hamza", "Chotiyay", "BariAy", "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P", "Q",
                  "R", "U", "V", "W", "Y"]
    image = prepare(array)

    prediction = model.predict([image])
    prediction = list(prediction[0])
    return Categories[prediction.index(max(prediction))]




def predictor(img):
    b=1
    path_file = ""
    if (int(b)>0):
        sign, x, y, w, h,isHandDetected  = yolo_object_detection.detectHand(img)
        # print(isHandDetected)
        font = cv2.FONT_HERSHEY_PLAIN
        # print(sign)
        if(isHandDetected != -1):
            letter = signPredictor(sign)
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img = cv2.putText(img, letter, (x, y +90), font, 7, (0, 255, 0), 5)
            path_file = ("static/%s.jpg" %uuid.uuid4().hex)
            cv2.imwrite(path_file, img)
            return json.dumps(path_file)
        else:
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            img = cv2.putText(img, "No Hand Found", (2, 200), font, 7, (0, 255, 0), 5)
            path_file = ("static/%s.jpg" % uuid.uuid4().hex)
            cv2.imwrite(path_file, img)
            return json.dumps(path_file)





# API
app = Flask(__name__)


# route
@app.route('/api/upload', methods=['POST'])
def upload():
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)

    processedImage = predictor(img)

    return Response(response=processedImage, status=200, mimetype="application/json")


# start Server
app.run(host="0.0.0.0", port=5000)
