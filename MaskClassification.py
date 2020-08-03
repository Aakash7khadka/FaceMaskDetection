import cv2
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
model=tf.keras.models.load_model(
    'Resnet_working',
    custom_objects=None, compile=True)
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture('Video.mp4')
labels_dict={0:'No Mask',1:'MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}

while True:
    sucess,img=video.read()
    faces=face_clsfr.detectMultiScale(img,1.3,5)
    # cv2_imshow(img)
    for x,y,w,h in faces:
        face_img=img[y:y+w,x:x+w]
     
        
        resized=cv2.resize(face_img,(244,244))
        

        input_=img_to_array(resized)
       
        input_=preprocess_input(input_)
       
        
        result=model.predict(np.array([input_]))
        label=np.argmax(result,axis=1)[0]
        print(max(result[0]))
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],1)
        cv2.putText(img,labels_dict[label]+':'+str(max(result[0])),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.imshow('Live',img)

        # out.write(img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# out.release()