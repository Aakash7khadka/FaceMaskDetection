# -*- coding: utf-8 -*-
"""Resnet_model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QWRnyu0zXWZ5XOq_CeUXyofNaAcgd54q
"""

from google.colab import drive
drive.mount("/content/drive")

base_path='drive/My Drive/Datasets/FaceMask'
paths=['with_mask','without_mask']

data=[]
labels=[]

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.layers import Input,Dense,AveragePooling2D,Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical

for path in paths:
  dir =os.path.join(base_path,path)
  for img in os.listdir(dir):
    image_path=os.path.join(dir,img)
    print(image_path)
    image=load_img(image_path,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)
    if path=='with_mask':
      labels.append(1)
    else:
      labels.append(0)

import numpy as np
data=np.array(data,dtype='float32')
labels=np.array(labels)
labels=labels.reshape(-1,1)
labels=to_categorical(labels)



aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data,labels,stratify=labels,random_state=0,test_size=0.2)
X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,stratify=y_train,random_state=0,test_size=0.1)

type(data)

base_model=ResNet50V2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))

base_model

x=base_model.output
x=AveragePooling2D(pool_size=(7,7))(x)
x=Flatten()(x)
x=Dense(128,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(64,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(2,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=x)

model.summary()

for layer in base_model.layers:
  layer.trainable=False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

H=model.fit(aug.flow(X_train,y_train),epochs=5,validation_data=(X_valid,y_valid))

pred=model.predict(X_test)

pred=np.around(pred)
pred=np.array(pred,dtype='int64')

from sklearn.metrics import f1_score
f1_score(y_test,pred,average='micro')

model.save('Resnet',save_format='h5')

model.evaluate(X_test,y_test)

t=X_test[0]
model.predict(np.array([t]))
"""
video_path=base_path+'/Video.mp4'
cascade_path=base_path+'/haarcascade_frontalface_default.xml'

!pip install opencv-python
import cv2

video=cv2.VideoCapture(video_path)
face_clsfr=cv2.CascadeClassifier(cascade_path)
from google.colab.patches import cv2_imshow

labels_dict={0:'No Mask',1:'MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}

frameSize = (720, 180)

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

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
        print(result)
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],1)
        cv2.putText(img,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        # cv2_imshow(img)

        out.write(img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

out.release()
"""
