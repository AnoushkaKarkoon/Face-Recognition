import cv2
import urllib
import numpy as np
from tensorflow.keras.models import load_model


classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("FACE-DETECT.h5")

url = "http://192.168.252.57:8080/shot.jpg"

def get_pred_label(pred):
    labels = ['Anoushka-ID-10001', 'Mahua-ID-10003', 'Mallika-ID-10002',
 'Swaptik-ID-10004']
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img,(100,100))
    img = img.reshape(1,100,100,1)
    img = img/255
    return img


ret = True
while ret:
    img_url = urllib.request.urlopen(url)
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv2.putText(frame,
                    get_pred_label(np.argmax(model.predict(preprocess(face)), axis=-1)[0]),
                    (200,500),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        
    cv2.imshow("cap",frame)
    if cv2.waitKey(30) == ord("q"):
        break
    
cv2.destroyAllWindows()