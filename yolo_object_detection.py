import cv2
import numpy as np
import glob
import random

def detectHand(img):
        # Load Yolo
        net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

        # Name custom object
        classes = ["hand"]

        # Images path
        images_path = img


        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))


        img = cv2.resize(images_path, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    # print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if (x < 0):
                        x = 0
                    if (y < 0):
                        y = 0
                    if (w < 0):
                        w = 0
                    if (h < 0):
                        h = 0

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        copyImage = []
        roi= []

        # print("boxes",len(boxes))
        if(len(boxes)!= 0):
            for i in range(len(boxes)):
                    x, y, w, h = boxes[i]
                    print(x,y,w,h)
                    roi = img[y:y + h, x:x + w]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    # cv2.rectangle(copyImage, (x, y), (x + w, y + h), color, 2)
                    # cv2.putText(copyImage, label, (x, y + 30), font, 3, color, 2)
            print("Hand is detected")
            print(roi)
            return roi, x, y, w, h, int(len(boxes))
        else:
            return -1,-1,-1,-1,-1,-1

