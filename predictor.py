import cv2
import tensorflow as tf
import numpy as np
import yolo_object_detection
import faceDetection







Categories = ["0","1","2","3","4","5","6","7","8","9","1Alif", "2Bay","3Pay","4Tay","5TTaay", "6TTSay","7Chay", "8Hay", "9Khay","10Dal","11Daal","12Zaal","13Ray","14Zay","15Saay","15Seen","17Saad","18Daad", "19Zuay","20Aeen","21Fay", "22Qaaf", "23Kaaf","24Gahf","25Laam","26noonGunna","27Wao","28Hamza","29Chotiyay","30BariAy","a","b","c","d","e","f","g","h","i","k","l","p","q","r","u","v","w","y"]


path = "TestDataSet/IMG20200822124452.jpg"


model = tf.keras.models.load_model(r"C:\Users\hira\PycharmProjects\SignDetection\detectSign.model")
IMG_SIZE = 28
def prepare(file):
    img_array = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    img_array = np.array(img_array) / 255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def signPredictor(array):
        image = prepare(array)
        prediction = model.predict([image])
        prediction = list(prediction[0])
        # print(prediction)
        print("Sign: ", Categories[prediction.index(max(prediction))])








img = cv2.imread(path)
imageCopy = np.copy(img)
imgCopy = faceDetection.detectFace(imageCopy)
a, b = imgCopy

if (int(b) > 0):
        print("face is detected")
        cv2.imshow("face", cv2.resize(a, (300, 300)))
        cv2.waitKey(0)
        sign = yolo_object_detection.detectHand(img)
        word = signPredictor(sign)
else:
        print("Face is not detected")



print("Done")
cv2.destroyAllWindows()