import cv2

def detectFace(img):
    face = cv2.CascadeClassifier('haarcascade_frontalface.xml')
    THIN_THRESHOLD = 10
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face = face.detectMultiScale(gray, 1.1, 3, minSize=(450, 450))

    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h + 30), (0, 0, 255), 3)
        # img[y:y + h + 30, x:x + w]
        if h < THIN_THRESHOLD or w < THIN_THRESHOLD:
            continue
    a = len(face)
    if(a>0):
        return  img,a
    else:
        return img,0