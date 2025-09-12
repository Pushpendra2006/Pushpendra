import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from keras import Sequential 
from sklearn.metrics import accuracy_score
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D 
from tensorflow.keras import datasets, layers, models
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train,x_test=x_train/255.0,x_test /255.0 #for scaling purposes

class_names = ['airplane','automobile','bird','cat','deer', 
               'dog','frog','horse','ship','truck']

model=Sequential([ 
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),    
    layers.MaxPooling2D((2,2)),     
    layers.Conv2D(64,(3,3),activation='relu'),     
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')   
]) 

model.summary() 
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

test_loss,test_acc=model.evaluate(x_test,y_test)*100
print(f"Test Accuracy:{test_acc}") 

plt.plot(history.history['accuracy'],label='accuracy') 
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy') 
plt.legend(loc='lower right')
plt.show() 

# Predict on a single image
img=x_test[0]
img=np.expand_dims(img,axis=0)
prediction=model.predict(img)
print("Predicted class:",class_names[np.argmax(prediction)])  